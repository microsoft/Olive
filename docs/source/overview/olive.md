# Olive

Olive is an easy-to-use hardware-aware model optimization tool that composes industry-leading techniques
across model compression, optimization, and compilation. Given a model and targeted hardware, Olive composes the best
suitable optimization techniques to output the most efficient model(s) for inference on cloud or edge, while taking
a set of constraints such as accuracy, latency and throughput into consideration.

Since every ML accelerator vendor implements their own acceleration tool chains to make the most of their hardware, hardware-aware
optimizations are fragmented. With Olive, we can:

Reduce engineering effort for optimizing models for cloud and edge: Developers are required to learn and utilize
multiple hardware vendor-specific toolchains in order to prepare and optimize their trained model for deployment.
Olive aims to simplify the experience by aggregating and automating optimization techniques for the desired hardware
targets.

Build up a unified optimization framework: Given that no single optimization technique serves all scenarios well,
Olive enables an extensible framework that allows industry to easily plugin their optimization innovations.  Olive can
efficiently compose and tune integrated techniques for offering a ready-to-use E2E optimization solution.

Contents
--------

The documentation is organized as follows:

- **OVERVIEW** provides an introduction to Olive and how it works.
- **GET STARTED** provides guidance to start with Olive.
- **EXAMPLES** provides E2E examples with Olive for various scenarios.
- **TUTORIALS** provides detailed instruction for using Olive.
- **API REFERENCE** provides an overview of the core Olive components.
