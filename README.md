# evaluation-gen-ai
Examples for evaluating generative AI use cases on Amazon Bedrock and Amazon SageMaker.

## Features

### 1. [Knowledge Base Evaluation](./knowledge_base_basic_ragas_evaluation.ipynb)
- Implements RAGAS framework for baseline testing of amazon Bedrock Knowledge bases
- Measures retrieval accuracy and relevance
- Evaluates context precision and faithfulness

### 2. [Knowledge Base Optimization](./optimize_knowledge_using_ragas_evaluation.ipynb)
- Use RAGAS to find the optimize knowledge base hyper parameters:
-- number of retreived answers
- Choice of generating model

### 3. Model Safety Assessment
- Integration with Bedrock Guardrails
- RAGAS safety metrics implementation
- Measure guardrail accuracy by analyzing tradeoffs between over-filtering (false positives) and under-filtering (false negatives).

### 4. Agent Evaluation Framework
- End-to-end agent testing
- Task completion verification
- Response quality measurement
- Performance benchmarking

## Contributing
Open an Issue or a Pull request.

## License
This project is licensed under the [LICENSE](LICENSE) file in the repository.
```