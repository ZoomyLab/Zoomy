nme: sms
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
    - sympy>=1.13.3
    - attrs
    - numpy
    - ipython
    - ipykernel
    - matplotlib
    - pytest
    - meshio
    - scipy
    - h5py
    - pyprog
    - mpi4py
    - petsc
    - petsc4py
    - pyvista
    - trame
    - trame-vtk
    - trame-vuetify
    - seaborn
    - jax
    - jaxlib
    - panel
    - watchfiles
    - pygments
    # - pyprecice
variables:
  PETSC_DIR: '/home/ingo/Git/petsc'
  PETSC_ARCH: 'linux-gnu'
  SMS: '/home/ingo/Git/sms'
  PYTHONPATH: ':/home/ingo/Git/sms'


