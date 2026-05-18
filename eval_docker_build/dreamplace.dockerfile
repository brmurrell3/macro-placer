# Build DREAMPlace v2: OLD ABI to match PyTorch 2.5.1 binary distribution
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    cmake bison flex git build-essential \
    libboost-all-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy pandas matplotlib pyunpack patool

WORKDIR /build
RUN git clone --recursive https://github.com/limbo018/DREAMPlace.git

WORKDIR /build/DREAMPlace
# Use OLD ABI (=0) to match PyTorch 2.5.1 official binaries.
# Pass via CMAKE_CXX_FLAGS so every subproject (Limbo, etc.) inherits.
RUN mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/dreamplace_install \
          -DPYTHON_EXECUTABLE=$(which python) \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
          -DCMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
          .. && \
    make -j$(nproc) && \
    make install

RUN ls -la /opt/dreamplace_install/dreamplace/ && \
    du -sh /opt/dreamplace_install

CMD ["/bin/bash"]
