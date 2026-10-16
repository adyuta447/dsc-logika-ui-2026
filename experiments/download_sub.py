import sys, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadSubmissionRequest
# Downloads a team submission file by its ref (from `kaggle competitions submissions`); the CLI has no command for this.
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as k:
    for ref, out in zip(sys.argv[1::2], sys.argv[2::2]):
        r = ApiDownloadSubmissionRequest(); r.submission_id = int(ref)
        b = requests.get(k.competitions.competition_api_client.download_submission(r).url, timeout=120); b.raise_for_status()
        open(out, "wb").write(b.content); print(out, len(b.content))
