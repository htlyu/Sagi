#!/bin/bash

# ide=code
ide=cursor
install_cmd="$ide --install-extension"
$install_cmd foxundermoon.shell-format
$install_cmd eamodio.gitlens
$install_cmd ms-python.isort
$install_cmd ms-python.black-formatter
