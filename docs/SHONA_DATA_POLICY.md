# NEO Shona speech data policy

## Required provenance per sample

Each record must retain source dataset, exact config/split, source license, license-review reference, transcript type, speaker identifier (pseudonymous), language/variety label, duration, sample rate, audio SHA-256 and local relative path.

## Permission gate

A dataset may enter a training run only after its terms have been checked for the intended purpose (research, internal development, commercial model training, redistribution). A dataset card that says `other`, an unclear license, or a source with conflicting upstream terms is blocked until written permission or a clear license basis is recorded.

## Privacy and consent

Do not collect private conversations, calls, voice notes or identifiable recordings without informed consent. Store speaker IDs as pseudonyms. Honour deletion/withdrawal requests where legally and contractually required. Avoid publishing raw voices or identity metadata.

## Quality tiers

- Gold: human-verified transcript and reviewed language/variety label.
- Silver: machine transcript with documented confidence and human spot-checking.
- Bronze: pseudo-label or canonical text alignment not yet checked against the audio.
- Quarantine: unclear rights, corrupted audio, duplicate, uncertain language or severe noise.

Only Gold should be used for final pronunciation evaluation. Silver may be used for training after filtering. Bronze must not be treated as ground truth.
