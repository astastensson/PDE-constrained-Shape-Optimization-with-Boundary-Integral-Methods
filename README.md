# PDE-constrained Shape Optimization with Boundary Integral Methods
A setup for solving shape optimization problems depending on a partial differential equation using boundary integral methods and the Python library JAX. The PDE in question is Laplaces equation with a Dirichlet boundary condition.
Any maximization problem where the goal function M depends on the solution u of Laplaces equations can be put into the setup.
The PDE is then solved using boundary integral methods, and M is optimized with gradient descent using automatic differentiation in JAX.
Some experiments and examples can be found in the Jupyter Notebooks.
The code was developed as part of a Bachelor thesis project by Rebecka Johansson and Asta Stensson for the Engineering Mathematics program at The Royal Institute of Technology (KTH) in spring 2024.

# PDE-villkorad formoptimering med randintegralmetoder
Ett ramverk för att slösa formoptimeringsproblem som beror på en partiell differentialekvation med randintegralmetoder och Pythonbiblioteket JAX. PDE:n i fråga är Laplaces ekvation med Dirichletvillkor.
Vilket maximeringsproblem som helst där målfunktionen M beror på lösningen u av Laplaces ekvation kan undersökas. 
PDE:n är löst med randintegralmetoder och M optimeras sedan med gradientstegning med hjälp av JAX automatiska differentiering.
Några experiment och exempel kan hittas i Jupyter Notebooksen.
Koden är utvecklad som ett kandidatexamensarbete av Rebecka Johansson och Asta Stensson för civilingenjörsprogrammet i teknisk matematik på Kungliga tekniska högskolan (KTH) under våren 2024.
