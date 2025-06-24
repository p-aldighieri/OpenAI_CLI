# OpenAI CLI Tool

This is a command-line tool to interact with OpenAI's models, including advanced options for context and reasoning settings.

## Usage

Run the script with the desired options. For example:

```bash
python openaipro.py "What is machine learning?" --context example.txt --model gpt-3.5-turbo --reasoning-effort medium
```

## Options

- **query**: The text query you want to send.
- **context**: Optional context text or filename.
- **model**: Model selection, e.g., `gpt-4`, `gpt-3.5`
- **reasoning-effort**: Level of reasoning used; options include `low`, `medium`, `high`.
- **max-tokens**: Maximum number of tokens in the response.
- **temperature**: Degree of randomness in the response.


## License

MIT License
