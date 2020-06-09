FROM rustlang/rust:nightly 

RUN apt update && apt install gfortran libopenblas-dev -y

COPY . /workspace
WORKDIR /workspace

RUN cargo build
RUN cargo test
