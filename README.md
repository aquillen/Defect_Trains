# Defect Trains that are launched by a phase discontinuity on the Boundary. 
Trains of Defects driven by a boundary in the complex Ginzburg Landau equation 

$\partial_t A = ( 1+ i \alpha) \Delta A + A - (1 + \beta) |A|^2 A$
with $A({\bf x}, t) \in {\mathbb C}$ and ${\bf x} \in {\mathbb R}^2$. 
Here $t$ is time and $\alpha, \beta \in {\mathbb R}$. 

In this example, the top boundary is Neumann and the left and right ones are Dirichlet. 
<img width="1800" height="1800" alt="CGLE_pin_v00__b1c__snap" src="https://github.com/user-attachments/assets/a1d45233-abb2-4983-8d70-46b978c22fc7" />

The PDE is integrated with a finite element method on a triangular mesh using NGSolve.
