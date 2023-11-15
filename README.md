# Δ-PINNs: physics-informed neural networks on complex geometries

Francisco Sahli Costabal, Simone Pezzuto, Paris Perdikaris

Physics-informed neural networks (PINNs) have demonstrated promise in solving forward and inverse problems involving partial differential equations. Despite recent progress on expanding the class of problems that can be tackled by PINNs, most of existing use-cases involve simple geometric domains. To date, there is no clear way to inform PINNs about the topology of the domain where the problem is being solved. In this work, we propose a novel positional encoding mechanism for PINNs based on the eigenfunctions of the Laplace-Beltrami operator. This technique allows to create an input space for the neural network that represents the geometry of a given object. We approximate the eigenfunctions as well as the operators involved in the partial differential equations with finite elements. We extensively test and compare the proposed methodology against traditional PINNs in complex shapes, such as a coil, a heat sink and a bunny, with different physics, such as the Eikonal equation and heat transfer. We also study the sensitivity of our method to the number of eigenfunctions used, as well as the discretization used for the eigenfunctions and the underlying operators. Our results show excellent agreement with the ground truth data in cases where traditional PINNs fail to produce a meaningful solution. We envision this new technique will expand the effectiveness of PINNs to more realistic applications

To cite

```
@article{sahli2024delta,
  title={$\Delta$-PINNs: Physics-informed neural networks on complex geometries},
  author={Sahli Costabal, Francisco and Pezzuto, Simone and Perdikaris, Paris},
  journal={Engineering Applications of Artificial Intelligence},
  volume={127},
  pages={107324},
  year={2024},
  publisher={Elsevier}
}
```
