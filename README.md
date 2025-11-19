# InferEdge Moss Integration for Pipecat

This integration enables Retrieval-Augmented Generation (RAG) in your Pipecat voice AI agents using InferEdge Moss vector search. The integration is authored and maintained by the Moss team (I work at Moss), so you can expect first-party support and regular updates as the platform evolves.

## Pipecat Compatibility

Tested with Pipecat v0.0.94. Please upgrade to this version (or newer) to ensure API compatibility with the snippets below.

## Overview

Moss is a vector database service that enables semantic search over your documents. When integrated with Pipecat, it automatically retrieves relevant context from your knowledge base based on user queries and augments the LLM's context with this information, enabling your voice AI agents to answer questions using your custom knowledge base.

## Installation

Install the package using uv:

```bash
uv sync
```

## Prerequisites

You'll need:

1. A Moss project ID and project key (get them from [InferEdge Moss](https://inferedge.com))
2. An existing Moss index with documents (or create one using the examples below)
3. Your API keys for other services (LLM, STT, TTS)

### Setting Up Environment Variables

Set your Moss credentials as environment variables:

```bash
export MOSS_PROJECT_ID="your-project-id"
export MOSS_PROJECT_KEY="your-project-key"
```

Or pass them directly when creating the client (see examples below).

## Quick Start

### 1. Creating a Moss Index (One-time Setup)

Before using Moss in your pipeline, you need to create an index and populate it with documents:

```python
import os
from src.client import MossClient, DocumentInfo

# Initialize the client (reads from MOSS_PROJECT_ID and MOSS_PROJECT_KEY env vars)
client = MossClient()

# Or pass credentials explicitly:
# client = MossClient(
#     project_id=os.getenv("MOSS_PROJECT_ID"),
#     project_key=os.getenv("MOSS_PROJECT_KEY")
# )

# Create documents
documents = [
    DocumentInfo(
        id="doc1",
        text="Pipecat is a framework for building real-time voice AI applications.",
        metadata={"source": "docs", "category": "introduction"}
    ),
    DocumentInfo(
        id="doc2",
        text="Frame processors are the building blocks of Pipecat pipelines.",
        metadata={"source": "docs", "category": "architecture"}
    ),
]

# Create an index with a specific embedding model
await client.create_index(
    index_name="pipecat-docs",
    documents=documents,
    model_id="text-embedding-3-small"  # or your preferred embedding model
)
```

### 2. Using MossRetrievalService in a Pipeline

Here's a complete example of integrating Moss retrieval into a Pipecat pipeline:

```python
import os
from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.runner.utils import create_transport
from pipecat.runner.types import RunnerArguments

# Import Moss integration
from src.client import MossClient
from src.retrieval import MossRetrievalService, RetrievalService

load_dotenv()

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # Initialize Moss client (reads from MOSS_PROJECT_ID and MOSS_PROJECT_KEY env vars)
    try:
        moss_client = MossClient()
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to initialize MossClient. "
            "Ensure `inferedge-moss` is installed and Moss credentials are set."
        ) from exc

    # Initialize services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
    )
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    # Configure Moss retrieval service (reuse the client instance)
    retrieval = MossRetrievalService(
        config=MossRetrievalService.Config(
            index_name=os.getenv("MOSS_INDEX_NAME", "pipecat-docs"),
            top_k=int(os.getenv("MOSS_TOP_K", 5)),
        ),
        params=RetrievalService.Params(
            system_prompt="Relevant passages from the Moss knowledge base:\n\n",
            add_as_system_message=True,
            deduplicate_queries=True,
            max_documents=int(os.getenv("MOSS_TOP_K", 5)),
        ),
        client=moss_client,  # Reuse the client instance
    )

    # Set up conversation context
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to a knowledge base. Use the retrieved context to answer questions accurately.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Build the pipeline
    # Note: retrieval service should be placed BEFORE the LLM
    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),  # Collect user messages
            retrieval,  # Retrieve relevant documents
            llm,  # Generate response with augmented context
            tts,
            transport.output(),
            context_aggregator.assistant(),  # Collect assistant messages
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
```

## How to Run the Example

1. **Create a `.env` file** that sets:
   - `MOSS_PROJECT_ID` - Your Moss project ID
   - `MOSS_PROJECT_KEY` - Your Moss project key
   - `MOSS_INDEX_NAME` - (optional, defaults to your index name)
   - `MOSS_TOP_K` - (optional, defaults to 5)
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `DEEPGRAM_API_KEY` - Your Deepgram API key
   - `CARTESIA_API_KEY` - Your Cartesia API key
   - Any transport-specific keys (e.g., Daily)

2. **Run the example**

   ```bash
   python examples/moss-retrieval-demo.py
   ```

   The script will start the Pipecat pipeline, verify Moss connectivity, stream STT into Moss-powered retrieval, and synthesize the LLM response with TTS.

## Configuration Options

### MossRetrievalService.Config

- `index_name` (required): Name of your Moss index
- `project_id` (optional): Moss project ID (can use env var `MOSS_PROJECT_ID`)
- `project_key` (optional): Moss project key (can use env var `MOSS_PROJECT_KEY`)
- `top_k` (default: 5): Number of documents to retrieve per query
- `auto_load_index` (default: True): Automatically load index before querying

### RetrievalService.Params

- `system_prompt` (default: "Here is additional context retrieved from memory:\n\n"): Prefix for retrieved documents
- `add_as_system_message` (default: True): Whether to add retrieved docs as a system message (vs user message)
- `deduplicate_queries` (default: True): Skip retrieval if the query hasn't changed
- `max_documents` (default: 5): Maximum number of documents to include in context
- `max_document_chars` (default: 2000): Maximum characters per document (longer documents are truncated)

## Advanced Usage

### Custom Client Instance

You can reuse a `MossClient` instance across multiple services:

```python
from src.client import MossClient
from src.retrieval import MossRetrievalService, RetrievalService

# Create a shared client (reads from env vars by default)
client = MossClient()

# Use it in multiple retrieval services
retrieval1 = MossRetrievalService(
    config=MossRetrievalService.Config(index_name="index1"),
    client=client,  # Reuse the client
)

retrieval2 = MossRetrievalService(
    config=MossRetrievalService.Config(index_name="index2"),
    client=client,  # Same client instance
)
```

### Managing Indexes

Use `MossClient` to manage your indexes:

```python
from src.client import MossClient, DocumentInfo, AddDocumentsOptions

client = MossClient()

# Create a new index with documents
documents = [
    DocumentInfo(
        id="doc1",
        text="Pipecat is a framework for building real-time voice AI applications.",
        metadata={"source": "docs", "category": "introduction"}
    ),
    DocumentInfo(
        id="doc2",
        text="Frame processors are the building blocks of Pipecat pipelines.",
        metadata={"source": "docs", "category": "architecture"}
    ),
    DocumentInfo(
        id="doc3",
        text="Moss retrieval service enables semantic search over your documents.",
        metadata={"source": "docs", "category": "retrieval"}
    )
]

# Create an index with a specific embedding model
created = await client.create_index(
    index_name="pipecat-docs",
    documents=documents,
    model_id="text-embedding-3-small"  # or your preferred embedding model
)
print(f"Index created: {created}")

# List all indexes
indexes = await client.list_indexes()
for index in indexes:
    print(f"Index: {index.name}")

# Get index information
index_info = await client.get_index("pipecat-docs")
print(f"Index has {index_info.document_count} documents")

# Add more documents to an existing index
new_docs = [
    DocumentInfo(
        id="doc4",
        text="New information about Pipecat features.",
        metadata={"source": "changelog"}
    )
]

result = await client.add_documents(
    index_name="pipecat-docs",
    docs=new_docs,
    options=None
)
print(f"Added documents: {result}")

# Delete documents
await client.delete_docs(
    index_name="pipecat-docs",
    doc_ids=["doc1", "doc2"]
)

# Delete an entire index
await client.delete_index("pipecat-docs")
```

### Custom System Prompt

Customize how retrieved documents are formatted:

```python
retrieval = MossRetrievalService(
    config=MossRetrievalService.Config(index_name="pipecat-docs"),
    params=RetrievalService.Params(
        system_prompt="Based on the following documentation:\n\n",
        add_as_system_message=True,
    ),
)
```

### Adding as User Message Instead

You can add retrieved documents as a user message:

```python
retrieval = MossRetrievalService(
    config=MossRetrievalService.Config(index_name="pipecat-docs"),
    params=RetrievalService.Params(
        add_as_system_message=False,  # Adds as user message instead
    ),
)
```

## How It Works

1. **Frame Interception**: `MossRetrievalService` intercepts `LLMContextFrame` or `LLMMessagesFrame` frames in the pipeline
2. **Query Extraction**: It extracts the latest user message from the conversation context
3. **Document Retrieval**: It queries your Moss index using semantic search
4. **Context Augmentation**: Retrieved documents are formatted and added to the LLM context
5. **Frame Forwarding**: The augmented context is passed downstream to the LLM

The retrieval happens automatically for every user message, ensuring your LLM always has access to relevant context from your knowledge base.

## Pipeline Placement

**Important**: Place the `MossRetrievalService` **before** the LLM service in your pipeline:

```python
pipeline = Pipeline([
    transport.input(),
    rtvi,
    stt,
    context_aggregator.user(),  # Collect user input
    retrieval,  # ← Must be before LLM
    llm,  # ← LLM receives augmented context
    tts,
    transport.output(),
    context_aggregator.assistant(),  # Collect assistant messages
])
```

## Error Handling

The service includes built-in error handling:

- If retrieval fails, an `ErrorFrame` is pushed upstream
- The original frame continues downstream, so the conversation isn't interrupted
- Errors are logged for debugging

## Best Practices

1. **Index Size**: Keep your indexes focused and well-organized
2. **Document Quality**: Ensure documents are well-formatted and contain relevant information
3. **Top-K Tuning**: Adjust `top_k` based on your document size and query complexity
4. **Metadata**: Use metadata to filter or provide context about document sources
5. **Deduplication**: Keep `deduplicate_queries=True` to avoid redundant retrievals
6. **Document Length**: Use `max_document_chars` to limit context size and prevent token overflow

## Troubleshooting

### Import Errors

If you see `RuntimeError: inferedge-moss is required`, install dependencies:

```bash
uv sync
```

### Authentication Errors

Ensure your `MOSS_PROJECT_ID` and `MOSS_PROJECT_KEY` are set correctly:

```bash
export MOSS_PROJECT_ID="your-project-id"
export MOSS_PROJECT_KEY="your-project-key"
```

### No Documents Retrieved

- Verify your index name is correct
- Check that the index has documents: `await client.get_index("your-index")`
- Ensure `auto_load_index=True` or manually call `await client.load_index("your-index")`

### Documents Not Relevant

- Adjust `top_k` to retrieve more/fewer documents
- Improve document quality and formatting
- Consider using better embedding models when creating the index
- Check that your documents are properly indexed and loaded

## License

This integration is provided under a permissive open source license (BSD-2 or equivalent).

## Support

This integration is maintained by the Moss team. For issues specific to this integration, please open an issue in this repository. For general Pipecat questions, join the [Pipecat Discord](https://discord.gg/pipecat).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
