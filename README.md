# Call Me If You Get Lost (CMIYGL)

CMIYGL is a voice-first navigation assistant for callers who need help figuring out where they are or getting to a destination. It accepts inbound phone calls and provides spoken directions.

## Requirements

- Python 3.11 or newer
- Git
- A Twilio Voice-enabled number
- Credentials for the configured speech, language-model, and mapping providers

## Setup

In an activated Python virtual environment, install the project:

```sh
python -m pip install -e .
```

Copy `.env.example` to `.env` and set the required provider credentials and configuration. Keep `.env` private.

## Run

Check configuration readiness:

```sh
python -m cmiygl.realtime
```

Start the phone server:

```sh
python -m cmiygl.realtime.server
```

For inbound calls, expose the server through a public HTTPS endpoint and set the Twilio number's Voice webhook to `https://<public-host>/twilio/voice` using `POST`. The default local server port is `8770`.

## Tests

Run the test suite from the repository root:

```sh
python -m unittest discover -s . -t .. -p "test*.py"
```
