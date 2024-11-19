# Evaluation-Gen-AI
Examples for evaluating generative AI use cases on Amazon Bedrock and Amazon SageMaker.

## Features

### 1. [Evaluating Amazon Bedrock Knowledge Base using RAGAS](./knowledge_base_basic_ragas_evaluation.ipynb)
- Implements RAGAS framework for baseline testing of amazon Bedrock Knowledge bases
- Measures retrieval accuracy and relevance
- Evaluates context precision and faithfulness

### 2. [Optimizing Amazon Bedrock knowledge Base using RAGAS](./optimize_knowledge_using_ragas_evaluation.ipynb)
- Use RAGAS to find optimal query time parameters for knowledge bases
-- number of retreived answers
-- Choice of generating model

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