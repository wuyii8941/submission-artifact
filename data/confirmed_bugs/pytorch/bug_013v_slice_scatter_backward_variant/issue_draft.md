# #<anon>: slice_scatter backward gradient corruption (variant)

- **Issue**: <upstream issue, anonymized for review>
- **Status**: Closed / linked variant
- **Related**: #<anon> (bug_013), #<anon> (bug_007)
- **Root Cause**: stride-0 broadcasting path triggered when slice_scatter's overwritten region excludes index 0, causing incorrect gradient scaling in Inductor's backward codegen.
- **Distinction from #<anon>**: #<anon> zeroes the gradient when the overwritten slice *includes* index 0 (constant-fold at coordinate 0). #<anon> produces incorrect gradient scaling when the slice *excludes* index 0 (different stride-0 broadcasting path).
