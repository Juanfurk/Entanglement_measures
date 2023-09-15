# Entanglement_measures

This repository has been created to store the codes of my master thesis, "Quantum Benchmarking: entanglement measurments in a quantum computer", with some examples. The codes containing two protocols for the estimation of the Rényi entropy can be found at "entanglement_measures.py", while an example of the execution can be checked at "Example_TFM.ipynb".

I have added "tequila_entanglement_measures.py". It is basically the same code of "entanglement_measures.py" but translated from qibo to tequila. However, this code needs some functions from qiskit and qibo in order to work properly. You can also find 'Tequila_Tests.ipynb', where I explaned some of the work I have been doing during the implementation of the previous code.

I have added "qiskit_entanglement_measures.py". Again, it is basically the same code of "entanglement_measures.py" but translated from qibo to qiskit. However, it only contains the Local approach, as the main objective of this code is to run some tests on the real IBM Quantum Harware (and the Global approach is not feasible). You can find 'Qiskit_Tests.ipynb', which contains some explanations during the implementation of the code.

I have added 'Qibo_Tests.ipynb', a notebook that contains some of the work I have been conducting post-TFM, like the p3-PPT criterian on Werner States.

Finally, the 'Post_TFM.pdf' contains everything I have been doing for this last couple of weeks. It is a much detailed explanation that also contain unfinished work with the path that may be follow in order to finish it.


