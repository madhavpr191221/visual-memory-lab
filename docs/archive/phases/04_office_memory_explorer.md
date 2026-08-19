# Phase 4: Office Memory Explorer

## Objective

Phase 4 turns the Phase 3 experiment into a usable local application without hiding the experiment behind a chatbot. A user can ask an office question or provide a reference image, inspect the retrieved memories, see how strongly those memories agree on a place zone, and then choose whether a vision-language model should judge a small evidence set. The underlying 7-Scenes data is licensed by Microsoft Research for non-commercial use; the dataset files are not redistributed by this repository.

The main design rule is **evidence before answers**. Retrieval is useful on its own and stays local. Model judgment is optional, explicit, and constrained to selected public images.

## Technician example

Imagine a technician remembers a workstation near a window but not its exact location. They ask:

> Where is the workstation beside a window?

The application embeds the question with the same frozen CLIP ViT-B/32 model used to build the memory. It compares that vector with all 6,000 stored Office vectors and returns the ten exact nearest memories. If seven of those memories belong to the window-side workstation zone, the interface reports strong zone agreement and displays the requested three, five, or ten images.

This is already a useful answer: the technician can see the evidence and recognize the place. If the question instead requires visual judgment—for example, “Does the aisle appear obstructed?”—the technician may select up to five returned frames and explicitly approve cloud analysis. The model must answer from those frames, cite their observation IDs, state the evidence strength, and explain limitations. It cannot infer who moved something, an event outside the frames, or calendar time that the dataset never recorded.

The same workflow supports more than object finding:

- location: “Where is the desk beside the interior window?”
- context: “What is around the bookshelf workstation?”
- revisit planning: “Which memories give the clearest view of the monitors?”
- visible state: “Does the aisle appear obstructed in these views?”
- maintenance evidence: “Is visible damage present around this workstation?”
- comparison: “Do these retrieved views appear to show the same desk?”
- object recall: “Show places where a desk phone is visible.”

## System flow

### 1. Load the frozen research artifacts

At application startup, FastAPI loads:

- the 6,000-frame training memory index;
- the 4,000-frame held-out query index;
- the seven curated Office zone definitions and assignments;
- the Phase 3 aggregate and per-query evaluation results;
- one CLIP encoder matching the stored embedding model and revision.

These resources are loaded once during the application lifespan. They are not rebuilt for each request.

### 2. Run local retrieval

For a text question, CLIP converts the text into one normalized vector. For an uploaded PNG or JPEG, CLIP converts the decoded image into one normalized vector. The `NumpyMemoryStore` then performs exact similarity search over every stored training embedding.

The backend always retrieves ten results because place-zone agreement is calculated over a stable evidence set. The interface can display three, five, or all ten without changing that calculation.

Image uploads are limited to PNG or JPEG files under 10 MB. They are decoded in memory and are not written to disk.

### 3. Summarize place agreement

Each stored training observation has a frozen Phase 3 zone assignment. The service counts the zones among the ten retrieved memories:

- **strong:** at least seven results support the leading zone;
- **moderate:** four to six results support one unique leading zone;
- **mixed:** weaker support or a tie.

This is an interpretable retrieval summary, not a new VLM call. It tells the user whether the returned images consistently point to one office area.

The interface also says that calendar time is unavailable. Sequence and frame numbers are experiment identifiers, not real dates.

### 4. Let the user inspect evidence

Every result includes its image, rank, CLIP score, observation ID, sequence, frame, and curated zone. Absolute filesystem paths are never returned by the API. Images are served only after an observation ID is resolved through a loaded index and its path is verified to remain inside that index's image root.

The result cards are the primary product. A generated paragraph is not required for simple recall or navigation.

### 5. Make cloud analysis an explicit second action

Normal search never calls OpenAI. The Analyze button is available only when `OPENAI_API_KEY` is configured, and pressing it first opens a confirmation panel. The user chooses between one and five evidence frames.

After confirmation, the backend sends the question and selected public frames to the configured model, currently `gpt-5.6-terra`. An image-based question also sends the uploaded reference image. The response follows a strict schema:

- question family;
- whether the evidence supports an answer;
- answer;
- evidence citations containing observation IDs and claims;
- low, medium, or high evidence strength;
- limitations.

The backend rejects citations to observation IDs that were not supplied. A supported answer must cite at least one selected observation.

Text-query judgments over public images are cached using the model, prompt version, schema, question, and image hashes. Uploaded-image judgments are never cached because the query image may be personal.

## Evidence Lab

The application contains a separate Evidence Lab so the polished search experience does not conceal model failures.

### Evaluation page

This page displays the frozen Phase 3 place-retrieval measurements:

- query count;
- strict and relaxed coverage;
- hit@1, hit@5, and hit@10;
- median and 90th-percentile pose error;
- per-sequence results.

Strict relevance means a stored frame is within 0.25 m and 30 degrees of the query pose. Relaxed relevance uses 0.50 m and 30 degrees. Coverage reports whether any qualifying memory exists, which separates retrieval failure from missing reference coverage.

### Failure browser

The browser pages through the 4,000 held-out queries and supports outcome tags such as strict top-1 success, rescued at five, miss at ten, uncovered, large translation error, and large rotation error. Each detail page shows the held-out query beside all ten retrieved memories with their similarity and physical pose errors.

### Zone browser

The zone pages expose the frozen semantic place vocabulary, stable landmarks, assignment counts, and representative memories. These labels help people interpret the retrieval space; they are not claimed as architectural ground truth.

## Architecture

The implementation deliberately keeps replaceable boundaries:

- `memory_store.py` defines the storage contract and the current exact NumPy implementation;
- `ui_service.py` contains retrieval summaries, zone access, and evaluation browsing;
- `api_models.py` defines stable JSON response schemas;
- `api.py` handles HTTP validation, resource lifespan, safe image serving, and production static serving;
- `vlm_analysis.py` owns the optional cloud boundary, structured response, citation checks, and cache policy;
- `web/` contains the React and TypeScript application.

The current dataset is small enough that exact in-memory search is simple and correct. A vector database can replace `NumpyMemoryStore` later without changing the user-facing API, but adding one now would not improve the 6,000-item experiment.

## Run the application

Install Python and frontend dependencies, build the React application, and start the local server:

```powershell
uv sync
Set-Location web
npm install
npm run build
Set-Location ..
uv run visual-memory-lab serve-ui
```

Open `http://127.0.0.1:8000`. Without an API key, all local search and Evidence Lab features still work; only optional VLM analysis is disabled.

For frontend development, run `npm run dev` inside `web` and run `uv run visual-memory-lab serve-ui` from the repository root. Vite proxies `/api` requests to FastAPI on port 8000.

## Verification

The phase is checked with:

```powershell
uv run python -m pytest
Set-Location web
npm run test
npm run lint
npm run build
```

The tests cover local search contracts, upload validation, allowlisted image serving, zone and failure endpoints, explicit analysis availability, citation-shaped responses, public-text cache reuse, non-caching of image queries, and the UI confirmation boundary.

## What this phase does not claim

- It does not know real capture dates or answer “when” questions about this dataset.
- It does not identify people or reconstruct unseen events.
- Place-zone agreement is not proof that every retrieved frame is correct.
- VLM analysis is a bounded interpretation of selected images, not an autonomous agent.
- The Office dataset demonstrates place memory; it is not yet a live workplace deployment.

## Attribution

The interface presents results derived from Microsoft Research's 7-Scenes
dataset and uses OpenAI CLIP ViT-B/32. See the repository
[Third-Party Notices](../../../THIRD_PARTY_NOTICES.md) for the non-commercial
dataset terms, original 7-Scenes paper, CLIP paper, code license, and model card.
