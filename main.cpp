// Radix selection using Kokkos; loosely based on PyTorch's implementation
// https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k_radix_selection.cuh
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iomanip>

const unsigned int RADIX_BITS = 8;
const unsigned int RADIX_SIZE = 1 << RADIX_BITS;
const unsigned int RADIX_MASK = RADIX_SIZE - 1;

template <class T> struct find_result {
    T val;
    bool found;

    KOKKOS_INLINE_FUNCTION find_result() : val(0), found(false) {}

    KOKKOS_INLINE_FUNCTION find_result &operator+=(const find_result &src)
    {
        if (src.found) {
            found = src.found;
            val = src.val;
        }
        return *this;
    }
};

namespace Kokkos
{
template <class T> struct reduction_identity<struct find_result<T>> {
    KOKKOS_FORCEINLINE_FUNCTION static find_result<T> sum()
    {
        return find_result<T>();
    }
};
} // namespace Kokkos

int main(int argc, char *argv[])
{
    int N = 10;
    int L = 100000;
    int K = 10;

    Kokkos::ScopeGuard kokkos(argc, argv);

    Kokkos::View<float **> distances("distances", N, L);
    Kokkos::View<int **> indices("indices", N, L);
    Kokkos::View<int **> topk("out", N, K);

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    Kokkos::parallel_for(
        "shuffle", N, KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < L; j++) {
                distances(i, j) = j + 0.1f;
            }

            auto generator = random_pool.get_state();

            // Fisher–Yates shuffle
            for (int j = 0; j < L - 2; j++) {
                int k = generator.urand(j, L);
                float tmp = distances(i, j);
                distances(i, j) = distances(i, k);
                distances(i, k) = tmp;
            }

            random_pool.free_state(generator);
        });

    typedef Kokkos::View<int *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>
        ScratchBins;

    Kokkos::parallel_for(
        "radix_select",
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO)
            .set_scratch_size(
                0, Kokkos::PerTeam(ScratchBins::shmem_size(RADIX_SIZE)))
            .set_scratch_size(
                1, Kokkos::PerTeam(ScratchBins::shmem_size(RADIX_SIZE))),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            int k = K;
            unsigned int mask = 0, desired = 0;
            bool found = false;

            ScratchBins bins(member.team_scratch(0), RADIX_SIZE);

            for (int digit_pos = 32 - RADIX_BITS; digit_pos >= 0 && !found;
                 digit_pos -= RADIX_BITS) {
                Kokkos::single(Kokkos::PerTeam(member), [=] {
                    for (int j = 0; j < RADIX_SIZE; j++) {
                        bins(j) = 0;
                    }
                });

                member.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, L), [=](int j) {
                        unsigned int val =
                            reinterpret_cast<unsigned int &>(distances(i, j));
                        if ((val & mask) == desired) {
                            unsigned int digit = val >> digit_pos & RADIX_MASK;
                            Kokkos::atomic_inc(&bins(digit));
                        }
                    });

                member.team_barrier();

                for (int j = 0; j < RADIX_SIZE; j++) {
                    int count = bins(j);

                    if (count == 1 && k == 1) {
                        mask |= RADIX_MASK << digit_pos;
                        desired |= j << digit_pos;

                        found = true;
                        break;
                    } else if (count >= k) {
                        mask |= RADIX_MASK << digit_pos;
                        desired |= j << digit_pos;

                        break;
                    }

                    k -= count;
                }
            }

            find_result<float> res;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, L),
                [=](int j, find_result<float> &upd) {
                    unsigned int val =
                        reinterpret_cast<unsigned int &>(distances(i, j));
                    if ((val & mask) == desired) {
                        upd.found = true;
                        upd.val = distances(i, j);
                    }
                },
                Kokkos::Sum<find_result<float>>(res));

            Kokkos::parallel_scan(Kokkos::TeamThreadRange(member, L),
                                  [=](int j, int &partial_sum, bool is_final) {
                                      if (distances(i, j) <= res.val) {
                                          if (is_final && partial_sum < K) {
                                              topk(i, partial_sum) = j;
                                          }
                                          partial_sum++;
                                      }
                                  });
        });

    const auto topk_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), topk);

    const auto distances_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace(), distances);

    std::cout << "top-" << K << " indices:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << topk_mirror(i, j) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "top-" << K << " distances:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << distances_mirror(i, topk_mirror(i, j)) << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
