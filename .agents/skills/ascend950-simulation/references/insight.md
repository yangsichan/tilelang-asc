# Insight Compatibility and Validation

## Events and Flows

- Use an Insight version that parses intra-block B/E events under their original names. Support was
  validated in 26.2.0. Otherwise, report the version limitation; do not rename intra-block events to
  SET/WAIT, add copies, or remap display tracks.
- In the validated 26.2.0 version, top-level IDs on GET/RLS X events still do not populate slice.flag_id.
  Keep original-name events on hardware PIPE tracks and add B/E SET_FLAG/WAIT_FLAG anchors marked with
  `viewer_compatibility_anchor=true` and source information. Anchors are not executed flag instructions.
- Display both GET/RLS and anchors for 1 tick starting at raw completion; their semantic duration is
  zero. Place anchors on `PIPE / GET_BUF (compat)` and `PIPE / RLS_BUF (compat)` respectively. The s/t/f
  endpoints use the corresponding anchor's pid/tid and start time. Flows represent acquire/release
  scopes, not instruction latency or cross-PIPE dependencies.
- Separate endpoint lanes avoid the observed `(flag_id, track_id)` first-slice depth lookup ignoring
  endpoint time, and long intervals on the same track obscuring markers. Changing thread_name alone
  does not change track membership; verify pid/tid.
- Use sunsetOrange (palette index 9) for slices and flows. Slice colors hash cname; flow colors hash
  cat. The script's `GET/RLS buffer scopes` category selects the same orange. Changing flow.cname alone
  has no effect.

## Actual Validation

Import into a fresh directory to avoid cached parsing databases. Distinguish these validation levels:

1. JSON/raw: source-record coverage, PC/tick/core/PIPE, pairing IDs, and separate counts for real
   instructions and anchors.
2. Parser: original GET/RLS names, native intra-block intervals, and anchor flag_id associations with flows.
3. Pipeline APIs: compare time, lane, and depth for every endpoint through `unit/threadTraces` and
   `unit/flows`; separately check all flows through `flow/categoryEvents`. One successful node does not
   validate the full view.
4. GUI: actually click slices and inspect flows. Disclose when untested; API validation is not GUI acceptance.

At 1.65 GHz, 1 tick is about 0.606 ns. Integer-nanosecond parsing can quantize it to 0 ns, affecting
visibility and hit testing. Preserve 1 tick and report the limitation; do not silently widen markers.
For empty query results, check the version, request parameters, and data before concluding that events
are absent. Do not reuse old-version validation conclusions or extrapolate GEMM results to VF/MHA.

Click the GET/RLS compatibility SET/WAIT anchors to inspect flows. Do not claim that original-name X
slices already support direct association.
