#!/bin/bash

set -e

function install_ta_lib() {
    wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
    tar -xzf ta-lib-0.6.4-src.tar.gz
    cd ta-lib-0.6.4
    ./configure
    make
    make install

    # clean up
    cd ..
    rm -rf ta-lib-0.6.4
    rm ta-lib-0.6.4-src.tar.gz
}

install_ta_lib
