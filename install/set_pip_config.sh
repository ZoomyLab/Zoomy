#!/bin/bash
echo 'export ZOOMY_DIR="${PWD}"' >>         .sms-pip-env/bin/activate
echo 'export PYTHONPATH=":${PWD}"' >> .sms-pip-env/bin/activate
echo 'export JAX_ENABLE_X64=True' >>  .sms-pip-env/bin/activate



