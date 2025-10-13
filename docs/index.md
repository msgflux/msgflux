# ᯓ➤ **msgflux**

![logo](assets/logo.png){ width="300", .center}

**msgflux** is an open-source framework designed for building multimodal AI applications with ease and flexibility. Our mission is to seamlessly connect models from diverse domains—text, vision, speech, and beyond—into powerful, production-ready workflows.

``` bash
pip install msgflux
```

msgflux is built on four foundational pillars: **Privacy**, **Simplicity**, **Efficiency**, and **Practicality**.

- **Privacy first**: msgflux does not collect or transmit user data. All telemetry is fully controlled by the user and remains local, ensuring data sovereignty and compliance from the ground up.

- **Designed for simplicity**: msgflux introduces core building blocks—**Model**, **DataBase**, **Parser**, and **Retriever**—that provide a unified and intuitive interface to interact with diverse AI resources.

- **Powered by efficiency**: msgflux leverages high-performance libraries such as **Msgspec**, **Uvloop**, **Jinja**, and **Ray** to deliver fast, scalable, and concurrent applications without compromising flexibility.

- **Practical**: msgflux features a workflow API inspired by `torch.nn`, enabling seamless composition of models and utilities using native Python. This architecture not only supports modular design but also tracks all parameters involved in workflow construction, offering advanced **versioning and reproducibility** out of the box. 

In addition to the standard container modules available in *PyTorch*—such as **Sequential**, **ModuleList**, and **ModuleDict**—*msgflux* introduces a set of high-level modules designed to streamline the handling of **multimodal inputs and outputs**. These modules encapsulate common tasks in AI pipelines, making them easy to integrate, compose, and reuse.

The new modules include:

- **Agent**: A central module that orchestrates multimodal data, instructions, context, tools, generation schemas, and templates. It acts as the cognitive core of complex workflows.

- **Speaker**: Converts text into natural-sounding speech, enabling voice-based interactions.

- **Transcriber**: Transforms spoken language into text, supporting speech-to-text pipelines.

- **Designer**: Generates visual content from prompts and images, combining textual and visual modalities for tasks like image generation or editing.

- **Retriever**: Searches and extracts relevant information based on a set of input queries, ideal for grounding AI models in external knowledge.

- **Predictor**: A flexible module designed to wrap predictive models, such as those from scikit-learn or other machine learning libraries, enabling smooth integration into larger workflows.


For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
