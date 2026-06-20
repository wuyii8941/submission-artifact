# #180772: slice_scatter backward gradient corruption (variant)

- **Issue**: <upstream issue, anonymized for review>
- **Status**: Closed / linked variant
- **Related**: #180771 (bug_013), #180164 (bug_007)
- **Root Cause**: stride-0 broadcasting path triggered when slice_scatter's overwritten region excludes index 0, causing incorrect gradient scaling in Inductor's backward codegen.
- **Distinction from #180771**: #180771 zeroes the gradient when the overwritten slice *includes* index 0 (constant-fold at coordinate 0). #180772 produces incorrect gradient scaling when the slice *excludes* index 0 (different stride-0 broadcasting path).
