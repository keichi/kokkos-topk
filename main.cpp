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
    int N = 10000;
    int L = 10000;
    int K = 300;

    Kokkos::ScopeGuard kokkos(argc, argv);

    Kokkos::View<unsigned int **> data("data", N, L);
    Kokkos::View<unsigned int **> out("out", N, K);

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);

    Kokkos::parallel_for(
        "shuffle", N, KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < L; j++) {
                data(i, j) = j;
            }

            auto generator = random_pool.get_state();

            for (int j = 0; j < L - 2; j++) {
                int k = generator.urand(j, L);
                unsigned int tmp = data(i, j);
                data(i, j) = data(i, k);
                data(i, k) = tmp;
            }
        });

    typedef Kokkos::View<int *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>
        ScratchBins;

    Kokkos::parallel_for(
        "radix_select",
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO)
            .set_scratch_size(
                0, Kokkos::PerTeam(ScratchBins::shmem_size(RADIX_SIZE))),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            int k = K;
            unsigned int mask = 0, desired = 0;
            bool found = false;

            ScratchBins bins(member.team_scratch(0), RADIX_SIZE);

            for (int digit_pos = 32 - RADIX_BITS; digit_pos >= 0 && !found;
                 digit_pos -= RADIX_BITS) {
                // std::cout << "mask=" << std::hex << std::setfill('0')
                //           << std::setw(8) << mask << " desired=" <<
                //           std::setw(8)
                //           << desired << std::dec << std::endl;

                Kokkos::single(Kokkos::PerTeam(member), [=] {
                    for (int j = 0; j < RADIX_SIZE; j++) {
                        bins(j) = 0;
                    }
                });

                member.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, L), [=](int j) {
                        if ((data(i, j) & mask) == desired) {
                            unsigned int digit =
                                data(i, j) >> digit_pos & RADIX_MASK;
                            Kokkos::atomic_inc(&bins(digit));
                        }
                    });

                member.team_barrier();

                // std::cout << "digit_pos=" << digit_pos << " bins=";
                // for (int j = 0; j < RADIX_SIZE; j++) {
                //     std::cout << bins(j) << ", ";
                // }
                // std::cout << std::endl;

                for (int j = 0; j < RADIX_SIZE; j++) {
                    int count = bins(j);

                    if (count == 1 && k == 1) {
                        mask |= RADIX_MASK << digit_pos;
                        desired |= j << digit_pos;

                        found = true;
                        break;
                    } else if (count >= k) {
                        // std::cout << "bin #" << j << " contains " << k
                        //           << "-th item" << std::endl;

                        mask |= RADIX_MASK << digit_pos;
                        desired |= j << digit_pos;

                        break;
                    }

                    k -= count;
                }
                // std::cout << std::endl;
            }

            find_result<unsigned int> res;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, L),
                [=](int j, find_result<unsigned int> &upd) {
                    if ((data(i, j) & mask) == desired) {
                        upd.found = true;
                        upd.val = data(i, j);
                    }
                },
                Kokkos::Sum<find_result<unsigned int>>(res));

            // std::cout << "found kth item=" << res.val << std::endl;

            Kokkos::parallel_scan(Kokkos::TeamThreadRange(member, L),
                                  [=](int j, int &partial_sum, bool is_final) {
                                      if (data(i, j) <= res.val) {
                                          if (is_final && partial_sum < K) {
                                              out(i, partial_sum) = data(i, j);
                                          }
                                          partial_sum++;
                                      }
                                  });
        });

    // const auto out_mirror = Kokkos::create_mirror_view_and_copy(
    //     Kokkos::DefaultHostExecutionSpace(), out);

    // std::cout << "top-k items=";
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < K; j++) {
    //         std::cout << out_mirror(i, j) << ", ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
