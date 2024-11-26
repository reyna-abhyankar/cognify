.. _cognify_tutorials_interface:

***********************
Programming Interfaces
***********************

Cognify supports several workflow programming models, including ``LangChain`` and ``DSPy``.

We also offer a simple *Cognify programming interface* for you to write workflows from scratch or manually port your existing Python programs.

In this section, we explain each of these programming interfaces using a unified example.

The example workflow we use is a **Math Problem Solver** involving two agents called in sequence:

1. **Math Interpreter Agent**: This agent analyzes the problem and form a math model.

2. **Solver Agent**: This agent computes the solution by solving the math model generated in the previous step.

.. note::

   The complete code for this example is also available at `<https://github.com/WukLab/Cognify/tree/main/examples/math>`_.


.. toctree::
   :maxdepth: 1

   program
   dspy
   langchain
