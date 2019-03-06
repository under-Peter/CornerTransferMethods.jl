# CornerTransferMethods.jl

requires `TensorNetworkTensors.jl`

## Basics

This package implements Corner-Transfer-Algorithms (*CTM*) for
infinite Projected Pair Entangled States (*iPEPS*).

The input of the algorithm is a rank-4 tensor `A_ijkl` that could represent a spin in a classical 2d partition function or the norm of a *iPEPS*.
Graphically that would be represented as

        k
        |
    l--[A]--j
        |
        i


The problem we're solving is this:
Given a tensor `A` which is used to build an infinite grid, as in

       |    |    |    |    |    |    |    |    |
    --[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--
       |    |    |    |    |    |    |    |    |
    --[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--
       |    |    |    |    |    |    |    |    |
    --[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--
       |    |    |    |    |    |    |    |    |
    --[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--
       |    |    |    |    |    |    |    |    |
    --[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--[A]--
       |    |    |    |    |    |    |    |    |

How can we construct the _effective environment_ of a particular
tensor?

In CTM-methods, we choose an ansatz for the environment consisting of tensors `C_` and `T_` that are connected with legs of dimension χ (`==` or `||` below)

    [C1]==[T1]==[C2]
     ||     |    ||
    [T4]--[ A]--[T2]
     ||     |    ||
    [C4]==[T3]==[C3]

and _iteratively_ improve the tensors `C_` and `T_` which represent
infinite corners and half-infinite columns/rows respectively as e.g.

       ||       |    |    |    |    |    |    |
    --[T2] = --[A]--[A]--[A]--[A]--[A]--[A]--[A]-⋯
       ||       |    |    |    |    |    |    |

until they are represent the best approximation of the environment given the ansatz-bond dimension χ.

## Notation

The bulk-tensors `A_ijkl` are always oriented as in

        k
        |
    l--[A]--j
        |
        i

i.e. the first index points down and the subsequent indices are counter-clockwise around the tensor.

Corner tensors `C1_ij`, `C2_ij`, `C3_ij` and `C4_ij` are oriented as

    [C1]==j   i==[C2]        i     j
     ||           ||        ||     ||
     i             j    j==[C3]   [C4]==i

i.e. moving clockwise in the environment, the first index encountered is the first index of the corner.

For the row- and column-tensors `T1_ijk`,`T2_ijk`,`T3_ijk`,`T4_ijk`, the indices are oriented as

    i==[T1]==k       i         j         k
        ||          ||        ||         ||
        j       j==[T2]   k==[T3]==i    [T4]==j
                    ||                   ||
                     k                   i
i.e. the again following the sequence when traversing the environment clockwise.

## Algorithms
Depending on the symmetry of `A`, there are different algorithms implemented:

### No Symmetry
The algorithm chosen for this situation is the _directional variant_ of *CTMRG*,
as described in [Simulation of two-dimensional quantum…](https://arxiv.org/abs/0905.3225), page 2f.
### Rotational Symmetry

If `A_ijkl = A_jkli`, i.e. `A` is rotationally symmetric,
we can restrit the environment ansatz to

    [ C]==[ T]==[C*]
     ||     |     ||
    [ T]--[ A]--[T*]
     ||     |     ||
    [C*]==[T*]==[ C]

Updating is then done using and enforcing this symmetry,
including on the tensor-level where `C_ij ≡ C_ji` and `T_ijk ≡ T_kji`.

The algorithm chosen here is 



### Hermitian w.r.t left/right and up/down

If `A_ijkl  ≡ A*_kjil ≡ A*_ilkj`, i.e. `A` is Hermitian w.r.t to both the left-right and up-down indices,
we can restrict the environment ansatz to

    [ C]==[T1 ]==[ C*]
     ||     |     ||
    [T2]--[ A ]--[T2*]
     ||     |     ||
    [C*]==[T1*]==[ C ]

The algorithm chosen for this situation is the _simplified one-directional 1D method_,
as described in [Exploring corner transfer matrices…](https://arxiv.org/abs/1112.4101),
page 9ff.

This can be used for e.g. _imaginary time evolution_ to find ground states of 1D systems if you can provide an MPO-Tensor `A` that satisfies the restrictions above.
