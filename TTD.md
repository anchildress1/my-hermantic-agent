# TTD

- [x] Move db out of public
- [ ] separate llm from cli

## Memory TTD

- [x] Add `memory_events` audit table
- [x] Add soft-delete for memories (`deleted_at`)
- [x] Disable manual memory mutation commands in CLI flow
- [x] Add explicit remember-intent priority boost (high importance)
- [x] Revive exact duplicates by refreshing access and boosting importance
- [x] Add event retention policy for `memory_events` (time-based pruning)
- [ ] Add migration for future embedding-dimension change workflow
- [ ] Add relevance-regression fixtures with tolerance bands for extraction behavior
- [x] Add tombstone lifecycle policy (merge/update/delete reconciliation with full audit)
