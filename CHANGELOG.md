# CHANGELOG

<!-- version list -->

## v3.17.0 (2026-09-07)

### Bug Fixes

- **streaming**: Sort CSC columns when pixels arrive out of raster order
  ([`4c0b697`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4c0b69731b2c2bf1855a7244b1d541c165edb1b3))

### Features

- **mobility**: Build the sibling tables out of core
  ([`103353b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/103353badf63f865b05835933231ff6bde6fe5d2))


## v3.16.0 (2026-09-07)

### Bug Fixes

- **mobility**: Refuse a grid table that will not fit in memory
  ([`1c92b1b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/1c92b1b1ecf8c33bfdf3b9c60539b66894495f3e))

### Features

- **mobility**: Bin a TDF point cloud onto a common mobility grid
  ([`c14ccbf`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/c14ccbfd04d0c883f5a5331cff32b247ef739494))


## v3.15.0 (2026-09-06)

### Bug Fixes

- **msms**: Keep isomer precursors apart, and label features intrinsically
  ([`a7d61f5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a7d61f500789146d0a4752c36a307f44c57e3b09))

### Documentation

- **msms**: Place the demultiplexed table against the prior art
  ([`9f86963`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/9f869631893f5869253f20d938aff2e563f6d787))

### Features

- **msms**: Split a PASEF frame into per-precursor spectra
  ([`3afca1b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/3afca1bd0a42c2947aeda467fcd91e94db2e0e8b))

- **schema**: Name the demultiplexed table in the metadata block
  ([`fce3fc8`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/fce3fc8e0c477f3a5e39c3b65f4034268182adcf))


## v3.14.0 (2026-09-06)

### Features

- **msms**: Record the fragmentation of an MS/MS acquisition
  ([`dc01af9`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/dc01af998c638835364cfc81bf748ceaaa81cea7))

### Refactoring

- **schema**: Group the instrument fields out of _build_ms_analysis
  ([`f831e1d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f831e1d918a2b52a2fc01616a31db2f7e4255b1f))


## v3.13.2 (2026-09-06)

### Bug Fixes

- **converters**: Store the mean, not the sum, as the 3D average spectrum
  ([`15dedf4`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/15dedf4edd7b818ff778898aa01f962c8969a104))


## v3.13.1 (2026-09-06)

### Bug Fixes

- **metadata**: Read stores through an extended-length path on Windows
  ([`ead3f68`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ead3f6854e3a96892ea026f337ae3d27dcc41f9e))


## v3.13.0 (2026-09-05)

### Features

- **mobility**: TDF mobility axis, mass-mobility heatmap and CCS
  ([`471320f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/471320f90cfda84e0ab7b291512646825f7d14e9))

### Refactoring

- **metadata**: Split the ion mobility block builder below the complexity threshold
  ([`bfeba4e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/bfeba4e569729d48d8d72c565bcb8c434653d514))


## v3.12.0 (2026-09-05)

### Features

- **imzml**: Read the ion mobility array and write the mobility-resolved table
  ([`6e610ec`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/6e610ecb9036c7eb305659db3849841f5f50e70e))


## v3.11.0 (2026-09-05)

### Bug Fixes

- **bruker**: Load the bundled Linux library under its shipped name
  ([`7077e71`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7077e71bdf212bc8a4d47771be1bda85881139e3))

- **bruker**: Read every TIMS scan of the right frame
  ([`76749d7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/76749d7cbb112940010e37719ed5dc7c8db2dec9))

### Features

- **metadata**: Record the mobility dimension and how TDF frames were summed
  ([`89070dd`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/89070dda674c4d521b16a24a4043268a04dc0d24))


## v3.10.1 (2026-09-02)

### Bug Fixes

- **solarix**: Open peaks.sqlite on UNC paths
  ([`21f4f3b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/21f4f3b54ea6a344739959b96799e4199ed7171f))


## v3.10.0 (2026-09-02)

### Documentation

- **handouts**: Add upstream handouts for spatialdata #1178 and zarr #4304
  ([`f34a5d0`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f34a5d0b0ed8525290a460fe41003434fd386ea8))

- **handouts**: Convert the remaining poetry instructions to uv
  ([`b3bbf51`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b3bbf511ab249d87b335d4667db3cbafda4edfab))

- **handouts**: Replace the stale poetry commands with uv
  ([`d4f1324`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d4f1324aa3016c42e3a039f65e6b410d2e74223d))

- **solarix**: Record the imzML cross-validation results
  ([`ee28d33`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ee28d337987eab4279aaacf8c695b28b525619f7))

### Features

- **readers**: Native solariX .d reader via peaks.sqlite
  ([`3baa953`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/3baa953ce07f31ffa2701c77a5f6b3ad122e7021))


## v3.9.0 (2026-09-01)

### Documentation

- List mzPeak in the supported formats
  ([`b09769c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b09769ced215c1d8dba442e2c4744a05fd7076bd))

### Features

- Experimental mzPeak reader
  ([`4bd6062`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4bd60621615d1f4a76bb0f5111b8516c5ba156b4))

### Testing

- Cover the mzPeak reader, extractor and imzML parity
  ([`23304e7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/23304e79d7e4bd50a73f66e30a09d20d1bcc4f81))


## v3.8.0 (2026-09-01)

### Features

- **resampling**: Recognise Waters, FT-ICR and exported timsTOF
  ([`7acb525`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7acb5255b9fe98fb5f47b93a26b096aa7ad51903))


## v3.7.5 (2026-09-01)

### Bug Fixes

- Raise the zarr floor to 3.1.6 so auto-sharding cannot pick 1 KB chunks
  ([`a86a94f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a86a94f82dcf0fedbc05c9874346ca0850fa0cdf))


## v3.7.4 (2026-08-31)


## v3.7.3 (2026-08-28)

### Bug Fixes

- Store non-numeric uns lists as JSON so read-back cannot segfault numpy 2.1-2.2
  ([`ca6f2a7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ca6f2a760ac1f3ad1c25edb727f7b4a0b2c4d74b))

### Documentation

- Add validation workflow notebook with Open in Colab badge
  ([`8d06236`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/8d0623678d549f302222c659dead0495c6af85e8))

- Clarify local (non-Colab) runs of the validation notebook
  ([`7e623c4`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7e623c40c43809bf931e2c8a127813d08a70b016))

- Floor numpy at 2.3 in notebook install to dodge string-array deepcopy segfault
  ([`ca900f5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ca900f51c9edd7145901f6b6ce4d139f5509288f))

- Keep Colab numpy untouched, work around numpy 2.1-2.2 StringDType deepcopy segfault in ROI cell
  ([`6f85aac`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/6f85aac3c676b2129a7de7044e1d6d0927a5db98))

- Make validation notebook run end to end
  ([`349cc0d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/349cc0d543aa9e4e56b7ccfebca4d44b32f1ba86))

- Pin pandas below 3 in notebook install to keep Colab runtime alive
  ([`177ce9a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/177ce9a81025128ba4249ee1a55e2fae05cd4098))


## v3.7.2 (2026-08-26)

### Bug Fixes

- Refuse silent peak drops in _map_mass_to_indices
  ([`f062233`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f062233b0dc7ab97f684d956bdd6cd1d939e5843))


## v3.7.1 (2026-08-26)

### Performance Improvements

- Cut per-spectrum recompute across the conversion hot loops
  ([`edfd4d1`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/edfd4d15ffc396e59fd3fc0469d7203f246211a6))


## v3.7.0 (2026-08-26)

### Features

- Record the resolved resampling plan in processing provenance
  ([`e7f24c0`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/e7f24c0839ba4f4e0abdfa06837bb186b608d757))


## v3.6.0 (2026-08-26)


## v3.5.1 (2026-08-26)

### Bug Fixes

- Convert imzML pixel size to micrometres using its declared unit
  ([`adeb252`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/adeb252ee819827538b3c98e86b6942f77eb589e))

### Build System

- Sync uv.lock with the 3.5.0 version bump
  ([`9b91660`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/9b916607dc0970ef9e40bc26004bdb7826b06e41))


## v3.5.0 (2026-08-25)

### Build System

- Move project metadata to PEP 621 [project] table
  ([`4410adf`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4410adf98e81355c221e4d578934e645ce8dad54))

- Swap the build backend from poetry-core to hatchling
  ([`b6cb013`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b6cb013a1c2cb810664dfc98079e171a0cc83576))

- Switch dependency management from Poetry to uv
  ([`dc14d0c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/dc14d0c670db782b791ef2248844daaab291c0e7))

### Documentation

- Update developer commands from poetry to uv
  ([`9694229`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/969422914891857d7cf77114d761507df7ecf3aa))

### Features

- Recognize Shimadzu imaging formats (.imdx, .kbd) with imzML guidance
  ([`21b9a56`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/21b9a562e54f99b02ce5310dcee99c3c40ad1103))


## v3.4.0 (2026-08-19)

### Features

- **resampling**: Recognise PHI ToF-SIMS instead of defaulting it to constant
  ([`ea21fa6`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ea21fa6ba695d3d4ab48b0e50f5e5a59600b2a25))


## v3.3.0 (2026-08-13)

### Bug Fixes

- **converters**: Stop the annotation hook pinning the CSC memmap open
  ([`6e2267d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/6e2267d5ba9c8a2db1fecd11b3ddb300a4b86167))

### Documentation

- **phi**: Say plainly which paths are untested, and invite the data
  ([`d70a93a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d70a93ab70a0db127bec7cc505b38b42851ec768))

### Features

- **readers**: Read PHI SmartSoft-TOF ToF-SIMS .raw data
  ([`bea94f9`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/bea94f9622fe3307b464e7815038b31ec3f2256c))


## v3.2.2 (2026-08-03)

### Bug Fixes

- **converters**: Keep a volume's pixel footprints two-dimensional
  ([`110d5e7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/110d5e78ba898af8db18620ce9fd3398704aa3bd))


## v3.2.1 (2026-08-03)

### Bug Fixes

- **converters**: Stop the PCS pre-scan sizing columns for spectra it skips
  ([`b8ef02a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b8ef02ae885c621fd6423a69058d3391cddc8264))

### Documentation

- **changelog**: Backfill the v3.0.0 entry the generator dropped
  ([`f831c28`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f831c286d36c95c6073c04d2bb640ed8bb6a50ac))


## v3.2.0 (2026-08-03)

### Documentation

- **output-format**: Retire the "pixel shapes are still 2D" limitation
  ([`c42b7b2`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/c42b7b2293d2bda99889d944bd0dbd2c6733ba1d))

### Features

- **converters**: Give a volume's pixel footprints their slice depth
  ([`7317792`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7317792a79f5fe55a9306edf2934cf5348ada494))


## v3.1.1 (2026-08-03)

### Performance Improvements

- **converters**: Make PCS the streaming default and drop the size threshold
  ([`291bdfd`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/291bdfd496afe003f747372f7dfa95d7ea85b402))


## v3.1.0 (2026-08-03)

### Bug Fixes

- **deps**: Declare click, tifffile and xarray as runtime dependencies
  ([`74cb21d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/74cb21da08aa14d4d3a6e46891205f9421c9209c))

### Documentation

- **tests**: Record that the whole imzML corpus is 1-based in x and y
  ([`d92e2a9`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d92e2a95ddbec90bc8995fb0fbdf7e164a8f8af2))

### Features

- **converters**: Give the 3D TIC volume a real z spacing
  ([`255c177`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/255c177c8816f3c7381ce89dc5306a6d2039398c))


## v3.0.3 (2026-08-03)

### Bug Fixes

- **cli**: Put every --help option in a category, and keep it that way
  ([`bbc7865`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/bbc7865b11dc44c726454a25802fa621f8224b1f))


## v3.0.2 (2026-08-03)

### Bug Fixes

- **converters**: Guard the streaming COO path against out-of-grid coordinates
  ([`2012287`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/2012287acbf56d74d72b985e20e1376116fd22d5))

### Documentation

- **handouts**: Mark H's remaining item blocked, not merely open
  ([`5ac757a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/5ac757a8a3abe99a85718a0b25fb7bd7a2e7b76b))

### Testing

- **converters**: Assert a named pixel's stored spectrum on all three write paths
  ([#150](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/150),
  [`3caad06`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/3caad062d3e63e73a86b5d5e8b964fb0b5212b9f))


## v3.0.1 (2026-08-03)

### Bug Fixes

- **converters**: Store the 3D TIC volume on the axes it declares
  ([`4f0cf46`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4f0cf468591230db6fddb65ce5278cb1b3718b59))

### Documentation

- Correct six false statements and document the preview entry point
  ([`7756a4b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7756a4bf2dc7066c66150d074fa7cf59808a4593))

- **handouts**: Add the loose-ends handout and refresh the index
  ([`36ea1fe`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/36ea1fe0c5fea05594ecb79b7ef9e904ac74e3c3))

- **handouts**: Record that the release-actor bypass does not exist
  ([`7ab456d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7ab456d7ea38a73b5a6b4d610e6e855cfb7ba856))

- **handouts**: Route the index to what is actually open
  ([`97a1928`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/97a1928a9ac51ae20609a43ad0fda0dc7562d245))

- **tests**: Record what the local imzML corpus can and cannot exercise
  ([`fcad673`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/fcad67361928dd918c6e396a2b035f5349cc0d74))

### Testing

- Repair the integration lane's streaming-converter fixtures
  ([`6247407`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/624740741cf00ddcb703f9407189d9fe68b17835))


## v3.0.0 (2026-08-02)

<!-- Hand-written. The generator rendered this section empty; the six commits
     below are `git log --no-merges v2.3.1..v3.0.0`. -->

### Bug Fixes

- **converters**: Drop the PCS path's phantom pixel rows, and refuse depth
  ([`87f287c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/87f287c4b10b78ab3acea1a91d0bd2959a7be25b))

- **converters**: Give the PCS store the root attrs and obs column it skipped
  ([`2c719d4`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/2c719d4fbbd8d12416d16bf67b0fc488a79f9441))

- **imzml**: Read file_mode and uuid from the file instead of by luck
  ([`d61a09f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d61a09fa0e6324fd26e0f2cc382e433dfe5d1e7e))

- **imzml**: Rebase z on the file instead of clamping z-1 at zero
  ([`76c18b8`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/76c18b88eb9945cda58e4c749260e54f1525101e))

- **metadata**: Refuse a dataset with no mass range instead of inventing one
  ([`3cae48b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/3cae48bceab94dceeadc50acb58e638447930f8f))

- **resampling**: Drop out-of-range peaks instead of folding them into edge bins
  ([`ef59115`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ef591158e0f44d9370bee3f1759e58270f65d090))

### Breaking Changes

- A PCS store's row indices change. Nothing migrates an existing one.

  The streaming-PCS path wrote one row per grid position where the other three
  write one per acquired spectrum. Acquisitions are polygon-shaped and the grid
  is their bounding box, so the corners came out as all-zero rows. On real
  `pea.imzML` that is 17,423 rows against 12,737 spectra, 4,686 of them empty,
  with a shapes polygon for each phantom. All four paths now agree at 12,737.

  This affects any store written by the streaming path with `use_csc=True`,
  which is chosen automatically for large datasets. Kept rows keep their *grid*
  index as `instance_id`, gaps and all -- only the row offsets compact. The TIC
  image stays full-grid, deliberately: it is a dense raster, not a per-pixel
  table, which is why `sum(TIC) == sum(X)` still holds.

  **Migration: reconvert.** Nothing rewrites an existing store in place, and
  code that indexes rows positionally will read the wrong pixel. Index by
  `instance_id` rather than by row offset and the change is invisible to you.

  To tell the two layouts apart on disk, count the root attrs: a store written
  before v3.0.0 by this path has 7, missing `coordinate_systems`,
  `format_specific_metadata` and `msi_dataset_info` (see the second commit
  above), and has no `obs/region_number` column. A v3.0.0 store has 10 and the
  column, on every path.

- `n_z > 1` is now refused on the streaming path rather than silently
  flattened. The PCS scatter indexes rows by `y * n_x + x` with no z term, so a
  multi-plane acquisition summed both planes onto one and returned success.
  Use the 2D or 3D converter, which the refusal message names.

- Stored values also change, without a row-layout change, for two inputs:
  resampling with a narrowed mass range (`--resample-min-mz` /
  `--resample-max-mz`) no longer piles the discarded part of every spectrum
  onto the first and last bins, and a dataset from which no spectrum yields a
  peak is now refused instead of being given an invented `(0.0, 1000.0)` mass
  range that also sized the common axis.


## v2.3.1 (2026-08-02)

### Bug Fixes

- **convert**: Let a reader's refusal out of the streaming size estimate
  ([`27ad37c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/27ad37c95e2b5e19850cb90d4d94be9118e6f46d))

- **imzml**: Remember a refused imzML instead of parsing it again
  ([`8ec81c7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/8ec81c76585d021f8b0cc774813eb41af914d97a))

- **imzml**: Validate pyimzml's parser state against the .ibd
  ([`f8e7009`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f8e7009d57f0893c935619a131b29e043dc46ea3))

### Documentation

- **fixtures**: State the corpus's measured compressed size
  ([`a3cb821`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a3cb821053d42ea8ef5dd436147a71a40278194c))

- **imzml**: Describe what the .ibd validator refuses
  ([`0995a12`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/0995a12e285d77c4ae0201f8a86a0fc3bcf5a865))

- **imzml**: Write down the pyimzml hazards that correctly get no code
  ([`21ab15a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/21ab15a95513c0864be6103198cfd9e85dd98879))

### Testing

- **fixtures**: Add a hand-authored imzML corpus that pyimzml did not write
  ([`540efd1`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/540efd18c6065bfcd222c366cb82079affdb28bc))

- **fixtures**: Assert the structural property each hand-authored pair exists for
  ([`362c604`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/362c60480784413323eb2530c29f664eb08bd090))

- **fixtures**: Guard the .gitignore negations and correct three stale notes
  ([`15d07da`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/15d07dae798397f6d171495ff38b7a97bf2e948b))

- **imzml**: Invert the two-precision characterisation onto the refusal
  ([`a48ff88`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a48ff880297d5c51c23417eae472b405b6995cfa))

- **imzml**: Resolve the .ibd in tests the way production resolves it
  ([`00a7de9`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/00a7de928184b1fd93184dcc561ae4ce666601a9))


## v2.3.0 (2026-08-01)

### Bug Fixes

- **converters**: Stop the streaming-PCS path inventing its own provenance
  ([`09a2db7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/09a2db767bc0ee52fb099499b6098572472f4adb))

### Features

- **imzml**: Take spectrum representation explicitly, as SCiLS does
  ([`5c7aa29`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/5c7aa294c0d64e1cab31d7828b24a379bf01cf04))


## v2.2.4 (2026-08-01)

### Documentation

- **resampling**: Source the SCiLS claims, and stop recommending a forbidden pairing
  ([`90df8cc`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/90df8cc837ed10e0df569a501d9e7550a038ad6f))


## v2.2.3 (2026-08-01)

### Bug Fixes

- **metadata**: Read the declared spectrum representation, don't infer it
  ([`d6c080e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d6c080e2b8caf1b7cfbbb402fe6809c7e6b83c9d))


## v2.2.2 (2026-08-01)

### Bug Fixes

- **resampling**: Stop handing unknown modalities MALDI-TOF resampling
  ([`f98bc3e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f98bc3e56407b0554e3e1df0842530a7e8db1164))


## v2.2.1 (2026-08-01)

### Bug Fixes

- **imzml**: Parse with ElementTree so unindented CRLF files load
  ([`b9fe017`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b9fe01726efbcf5270bebe49af10b6dbfdeb5da6))


## v2.2.0 (2026-08-01)

### Features

- **resampling**: Give tic_preserving a gap tolerance, as Cardinal and matter have
  ([`954c95d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/954c95d7f2955adeea03a65d048010a8e25b8e94))


## v2.1.0 (2026-08-01)

### Documentation

- **handouts**: Record the SCiLS baseline for resampling
  ([`35fc7e7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/35fc7e72aecd8c57494d9eaffae208f486739982))

### Features

- **readers**: Default the raw mass axis cap to SCiLS's 10 million bins
  ([`93eb229`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/93eb229371bd07ba5d43647fc3ea236c59ae3265))


## v2.0.4 (2026-08-01)

### Bug Fixes

- **resampling**: Stop dropping negative bins, and correct the docs
  ([`96938e3`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/96938e3c60b3331106ff14e06f027c6f2fcdc430))


## v2.0.3 (2026-07-31)

### Bug Fixes

- **streaming**: Route on the real bin count, not a guessed one
  ([`3ad7c0a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/3ad7c0abe21d286c7e49ef488e0a0f419fa3ee62))


## v2.0.2 (2026-07-31)

### Bug Fixes

- **streaming**: Widen CSR column indices and build the string indexes lazily
  ([`edf447a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/edf447a6a70868640238a9c6a2737f9ebb6c0b4c))


## v2.0.1 (2026-07-31)

### Bug Fixes

- **resampling**: Make tic_preserving actually preserve TIC
  ([`106f15e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/106f15e62c80d1ca107b48103178750c7b75cfb6))

### Documentation

- **handouts**: Mark the TIC finding as fixed
  ([`4c88657`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4c88657af0caf481c0c0cafba4900474d121b052))

- **handouts**: Report findings for the write-amplification investigation
  ([`6fd1798`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/6fd1798f940a140e7e356c8f07682bc20922e4cc))

### Performance Improvements

- **imzml**: Stream the processed-mode raw mass axis build
  ([`b8562b4`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b8562b43c38d8bc250c026c5353f8c31e51b5277))

### Refactoring

- **resampling**: Share one TIC-preservation rule between both paths
  ([`e2a2b58`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/e2a2b580445fc739360fe0029c3b04fe33ef7207))


## v2.0.0 (2026-07-31)

### Build System

- Drop Python 3.11 and upgrade the spatialdata cluster
  ([`0ef2263`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/0ef2263e981cd91427bb284563c596ce0f572f75))

### Documentation

- Correct the anndata pandas-3 fix status
  ([`d33460e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d33460e7d8d05d5facff3a9c5cfbdba86959de28))

- Note that the anndata escape hatches work in combination on pandas 3
  ([`54c0393`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/54c0393556fa39547e7ceaa0cf919bcd4b173e88))

- Record the real anndata 0.13 prerequisites in the pin comment
  ([`951319f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/951319fa349441334c6de0690346a47c06594f47))

- **handouts**: Refresh handout E after the #1055 rebase
  ([`87209aa`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/87209aaef1bc6021e59807df3199e2244bbbc598))

### Testing

- Make tests a regular package so site-packages cannot shadow it
  ([`f68efee`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f68efee18780b266ed7e2b424c0cff3ea7af3c65))

### Breaking Changes

- Python 3.11 is no longer supported. The minimum is now 3.12, required by spatialdata 0.8.0 and
  anndata 0.13.


## v1.27.3 (2026-07-30)

### Bug Fixes

- **ci**: Mark the complexity monitor's subprocess import as reviewed
  ([`bfe811b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/bfe811b5a71e169b67a0cdc42a1cf01f3f54706c))

### Build System

- Check out LF line endings on every platform
  ([`04879ab`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/04879ab2dd1bfddae2837f536062635f0804d570))

- Make pyproject.toml the single source of truth for pydocstyle
  ([`eecabb2`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/eecabb2537fcd4bd3c18e739c68fc9ce1f48af42))

### Documentation

- **handouts**: Record how the toolchain-hygiene lane resolved
  ([`41d9c62`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/41d9c62639574b42d16946059585cecb2a191c79))

### Refactoring

- **types**: Drop the two no-any-return errors from a full mypy run
  ([`f207f8d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f207f8d22c1adbb25b41c3582c22190a31273139))


## v1.27.2 (2026-07-30)

### Bug Fixes

- **cli**: Retire the --optimize-chunks pass that never ran
  ([`1d365c0`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/1d365c04c2f4f02ec7bbbd59b48c33c03e649b4e))

### Documentation

- **handouts**: Add handout F and record the branch sweep
  ([`c62afc9`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/c62afc9ad1ee18963b8bf651288d1a21e9832b14))

- **handouts**: Record how the optimize-chunks lane resolved
  ([`a868b41`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a868b41001d3b683f3b53112dc8d02963e5e2270))

### Testing

- **streaming**: Cover the encoding attrs the PCS layout hand-writes
  ([`9786359`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/978635921bee796b7821dd22af565ebb958612d6))


## v1.27.1 (2026-07-30)

### Bug Fixes

- **spatialdata**: Coerce pandas 3 string dtypes to object on write
  ([`86ffb1d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/86ffb1d5dd0af817b02502f4f3ba0bd121098d66))

### Documentation

- Add handouts for the four parallel workstreams
  ([`70be229`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/70be229a0ff7cf989b840da308a6b12cb76f3708))

- **handouts**: Fold in findings from spatialdata PR #1055
  ([`2d0851a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/2d0851a8c85df7888eec3afdc659c353d761772e))


## v1.27.0 (2026-07-30)

### Bug Fixes

- **cli**: Exit non-zero when conversion fails
  ([`1584f10`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/1584f106af07875b812102af36ef693bf8815c8c))

- **convert**: Write through an extended-length path when the store is deep
  ([`f9c0199`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f9c019938fbeab1133efeb0fdcb6722400036caf))

- **converters**: Derive bin count from axis physics for FT-ICR and Orbitrap
  ([`20e39de`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/20e39defa1b0e87e3278a54d65362a03027544d8))

- **converters**: Reject unknown resampling method and axis type values
  ([`4840b7d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4840b7da5670b7357738858a60eb1eedcfa2979c))

- **converters**: Retry Zarr metadata renames blocked on Windows
  ([`af19db6`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/af19db6b42a0cc9debf38e82a3f895f3597cb2dc))

- **resampling**: Align the ResamplingConfig reference_mz default with the CLI
  ([`de33301`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/de33301011d4e54af196e79326a728f33f122f7b))

- **resampling**: Emit Orbitrap and FT-ICR axes in ascending m/z order
  ([`40a191e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/40a191e4a1ed14a4ccabb335038b69b30819b0a0))

### Build System

- Make .flake8 the single source of truth for lint settings
  ([`e144f2d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/e144f2d3b626342ddd58d7986def874b5fe01dbb))

### Features

- **cli**: Add --version
  ([`b19f989`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b19f989ab7419ab1b4416b8c1431e61809fff50b))

### Refactoring

- **resampling**: Drop the dead axis-generator width and bins methods
  ([`c6bf58d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/c6bf58d5b2035e4576389bd2f3731d66870aa221))

- **resampling**: Drop the unimplemented LINEAR_INTERPOLATION method
  ([`23f8f76`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/23f8f7687273db7ad5be1eeb128a3bab90522894))


## v1.26.0 (2026-07-30)

### Documentation

- Add tutorial and resampling pages, publish the example notebook
  ([`45b3680`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/45b3680549d82a37e918bfc35ba5f924239dc04a))

### Features

- **tools**: Add thyra-example-data synthetic dataset generator
  ([`6566a77`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/6566a7721ce7384dc81b385e9dd06f3587527cdd))


## v1.25.5 (2026-07-09)

### Bug Fixes

- **deps**: Add upper bounds so pip install lands on a working set
  ([#102](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/102),
  [`0210b0e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/0210b0ed55199b81b7947e8313e48d82da933492))


## v1.25.4 (2026-06-22)

### Bug Fixes

- **streaming**: Write PCS images via spatialdata and fail loudly on error
  ([#100](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/100),
  [`5347a76`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/5347a76746406efa9fd838b1e673a3c10a5a68ec))


## v1.25.3 (2026-06-22)

### Bug Fixes

- **convert**: Revert RAM-aware auto-streaming threshold (#98)
  ([#99](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/99),
  [`8474134`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/847413481213c0a264b6ddd3c97ee170aea3bf41))


## v1.25.2 (2026-06-22)

### Bug Fixes

- **convert**: Make auto-streaming threshold RAM-aware
  ([#98](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/98),
  [`4e82aa3`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4e82aa3a890a211d585a7925723548ec4d392886))


## v1.25.1 (2026-06-22)

### Bug Fixes

- **deps**: Declare defusedxml as runtime dependency for Bruker reader
  ([#97](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/97),
  [`2bc4f86`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/2bc4f86d87ff1d32c7c0f735f1826be511cc0b6b))


## v1.25.0 (2026-06-12)

### Bug Fixes

- **streaming**: Clean up temp storage on every convert() exit path
  ([#96](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/96),
  [`c56f0bb`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/c56f0bbff491b704f64e907e93db49bd1a347a91))

### Documentation

- Credit Nepsis Scriptorium and place logotype on docs landing
  ([`b6e629e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b6e629e890b9857c7b372982c0eaf244d1bef7e7))

- Switch brand assets from PNG to SVG
  ([`344e152`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/344e1524dafeb78e0cdea5c642173210b98d8b2b))

### Features

- **spatialdata**: Centralize image chunk policy as the sharding seam (foundation)
  ([#95](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/95),
  [`59eb93b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/59eb93b8bacb2dd5f8bf9feabb25439ee9ea53dc))


## v1.24.0 (2026-06-01)


## v1.23.0 (2026-05-19)

### Bug Fixes

- **resampling**: Permissive timsTOF instrument-name match
  ([`2a200db`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/2a200db4571d8d3cf3d59a6a0fa108ab8906e6b9))

### Features

- **preview**: Add thyra.preview_msi metadata-only shim
  ([`2d06d1f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/2d06d1f41e2cec63387b9e73c50e713a44c2fb68))

- **preview**: Metadata-only mode for BrukerReader (skip SDK init)
  ([`8f23ef7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/8f23ef778f0084d4cf99f849a0f97120cd694f7e))

### Performance Improvements

- **preview**: Skip SUM(NumPeaks) scan in metadata_only mode
  ([`ac3f214`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ac3f21495c26c27138d185e7b08e30527b192f29))


## v1.22.0 (2026-05-06)

### Bug Fixes

- Align global coordinate system across image and shapes
  ([#93](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/93),
  [`36938e7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/36938e7facec910b9be095f60608931001f3e8a9))

### Documentation

- Document the coordinate-system contract
  ([#93](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/93),
  [`36938e7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/36938e7facec910b9be095f60608931001f3e8a9))

### Features

- Unified "global" coordinate-system contract for produced zarrs
  ([#93](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/93),
  [`36938e7`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/36938e7facec910b9be095f60608931001f3e8a9))


## v1.21.0 (2026-05-04)

### Bug Fixes

- Bruker region selection, pixel size, and size estimator bugs
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))

- Cast function_types dict keys to string for zarr serialization
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))

- Compute size estimator bins from real resampling axis
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))

- Drop empty pixel rows from obs when polygon != bbox
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))

- Prefer .mis Raster over BeamScanSize for pixel size
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))

### Features

- Accept .mis Area Names on --region; surface mapping
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))

### Testing

- Attach handler directly to module logger for log capture
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))

- Capture log via root logger to fix CI flake
  ([#92](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/92),
  [`f4177d5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f4177d5f13d9d01cc585f95f91d80ce73285d655))


## v1.20.3 (2026-03-24)

### Performance Improvements

- Vectorise _create_coordinates_dataframe with numpy instead of Python loop
  ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))

### Refactoring

- Clean up convert.py and align flake8 line length with black
  ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))

- Code quality improvements ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))

- Normalise resampling config to ResamplingConfig at init, remove isinstance branching
  ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))

- Remove duplicate _suppress_upstream_warnings from streaming_converter
  ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))

- Remove hardcoded path, clean up type ignores, delete dead integration test
  ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))

### Testing

- Add unit tests for MSIRegistry format detection and registration
  ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))

- Clean up test suite - remove dead tests, mark integration, add unit tests
  ([#86](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/86),
  [`7eeaf40`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7eeaf40a18d8463df678b49106420ff4d47f44eb))


## v1.20.2 (2026-03-21)

### Bug Fixes

- Compute bounding box from all points in polygon Area definitions
  ([#85](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/85),
  [`76741a5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/76741a5bf12c12b24ab7d62e450767078ff3d9db))


## v1.20.1 (2026-03-17)

### Bug Fixes

- Iterate actual frame IDs from MaldiFrameInfo instead of assuming sequential 1..N
  ([#83](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/83),
  [`91642cb`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/91642cbdca502933b3c76f04210f8c53dcc87e36))

- Pass per-region avg spectrum through COO path
  ([#83](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/83),
  [`91642cb`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/91642cbdca502933b3c76f04210f8c53dcc87e36))

- Pass per-region avg spectrum through COO path data structures
  ([#83](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/83),
  [`91642cb`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/91642cbdca502933b3c76f04210f8c53dcc87e36))


## v1.20.0 (2026-03-11)

### Features

- Store thyra_version in essential_metadata
  ([#82](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/82),
  [`7c309e2`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/7c309e215701d5b26bb85bc1282010077cc749c3))


## v1.19.0 (2026-03-10)

### Bug Fixes

- Extract region accumulation helpers to reduce complexity
  ([#81](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/81),
  [`159494e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/159494eb9cbdf518bfa69638142ab24ff130fbd6))

### Features

- Compute and store per-region mean spectrum for multi-region datasets
  ([#81](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/81),
  [`159494e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/159494eb9cbdf518bfa69638142ab24ff130fbd6))

- Per-region mean spectrum for multi-region datasets
  ([#81](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/81),
  [`159494e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/159494eb9cbdf518bfa69638142ab24ff130fbd6))


## v1.18.2 (2026-03-10)

### Bug Fixes

- Update Bruker SDK binaries to support TSF/MALDI on Linux
  ([#80](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/80),
  [`b629c0a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b629c0a81806c3e1865d31d7339ba833d7feb037))

### Documentation

- Add Bruker SDK license and attribution for bundled binaries
  ([#80](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/80),
  [`b629c0a`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b629c0a81806c3e1865d31d7339ba833d7feb037))


## v1.18.1 (2026-03-10)

### Bug Fixes

- Prevent duplicate PyPI publish on concurrent release runs
  ([`40884a6`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/40884a68f6106f43e8640004b8fc44840e0f257a))

### Documentation

- Add prominent documentation links to top of README
  ([`b58f27b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/b58f27b7bc2f80f1c0b95ae581540acf23e1225e))


## v1.18.0 (2026-03-10)

### Bug Fixes

- Correct changelog dates, dark mode, nav tabs, mypy version
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Dark mode admonitions, navigation polish, mypy version
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Docs review sweep - accuracy, mobile, and packaging fixes
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Final docs sweep - version gap note, obs columns, kwargs type
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Pin mkdocs <2 and change streaming default to auto
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Resolve tab visibility and layout issues in docs theme
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

### Code Style

- Brand admonition colours for note, tip, info, and warning
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Clean tab bar beneath gradient header
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

### Documentation

- Add MkDocs Material documentation site
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Complete API reference with metadata, resampling, and registry
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

- Comprehensive documentation overhaul
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))

### Features

- Apply brand colour scheme and typography to docs
  ([#79](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/79),
  [`a2d662c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/a2d662c59d9dfe67cd713ca095ca1997c3307687))


## v1.17.2 (2026-03-10)

### Bug Fixes

- Add region_number to obs in streaming and 3D converters
  ([`3dea7ca`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/3dea7caf6c07713694ae8daa95dff3d4b414133b))


## v1.17.1 (2026-03-09)

### Bug Fixes

- Store region info as JSON to preserve dict structure in zarr
  ([`49d674e`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/49d674e5e72a23c4700a6efdcb5721dd9ee87cb4))


## v1.17.0 (2026-03-09)

### Features

- Include area names from .mis file in region info output
  ([`2c34961`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/2c34961a51ecc9f12ab2e783136fc864022611d9))


## v1.16.0 (2026-03-04)

### Bug Fixes

- Correct multi-brain optical alignment for shared TIFF slides
  ([#78](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/78),
  [`77fed1d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/77fed1de8edc6129a9fb6908cfc3b928810529ec))

### Features

- Interactive dataset selection, grouped CLI help, resample default
  ([#78](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/78),
  [`77fed1d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/77fed1de8edc6129a9fb6908cfc3b928810529ec))

- Multi-brain alignment fix and CLI improvements
  ([#78](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/78),
  [`77fed1d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/77fed1de8edc6129a9fb6908cfc3b928810529ec))


## v1.15.1 (2026-02-26)

### Bug Fixes

- Code quality sweep - mypy, logging, asserts, zarr consolidation
  ([#77](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/77),
  [`fd51e6f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/fd51e6f8989617a664e9f911502e1d101dfa28f9))


## v1.15.0 (2026-02-25)

### Features

- Multi-region support, optical alignment fixes, and image scaling
  ([#76](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/76),
  [`80c0d0b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/80c0d0bf89d9cdf7420d6f0266c1448e8ecabece))

- Optical alignment transforms, multi-region support, and streaming fixes
  ([#76](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/76),
  [`80c0d0b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/80c0d0bf89d9cdf7420d6f0266c1448e8ecabece))

- Optical alignment, multi-region support, and streaming fixes
  ([#76](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/76),
  [`80c0d0b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/80c0d0bf89d9cdf7420d6f0266c1448e8ecabece))

### Refactoring

- Extract _resolve_imaging_bounds to reduce complexity
  ([#76](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/76),
  [`80c0d0b`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/80c0d0bf89d9cdf7420d6f0266c1448e8ecabece))


## v1.14.1 (2026-02-18)

### Bug Fixes

- Resolve all 287 mypy type errors across codebase
  ([#65](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/65),
  [`899d81f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/899d81f499fb1218e46fd94a7cf58b22c3768cdf))


## v1.14.0 (2026-02-13)

### Bug Fixes

- **ci**: Fix release workflow to properly detect and publish new versions
  ([`76f9bb0`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/76f9bb08ed26fccd99c5e7fb6bbdf1c53e3546ce))

### Documentation

- Update README with Waters .raw format support
  ([#73](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/73),
  [`d372d3c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d372d3cac6ad8d573eb9f80ef26c96acaf51900b))

### Features

- Add Waters .raw MSI reader with MassLynx native library support
  ([#73](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/73),
  [`d372d3c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d372d3cac6ad8d573eb9f80ef26c96acaf51900b))

### Refactoring

- Reduce cyclomatic complexity in main() and _scan_all_ms_spectra
  ([#73](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/73),
  [`d372d3c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d372d3cac6ad8d573eb9f80ef26c96acaf51900b))

### Testing

- Add comprehensive unit tests for Waters reader
  ([#73](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/73),
  [`d372d3c`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d372d3cac6ad8d573eb9f80ef26c96acaf51900b))


## v1.13.0 (2026-01-24)

### Features

- Add CLI support and tests for intensity threshold
  ([#71](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/71),
  [`ab1dfe4`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ab1dfe4f9fc9f81c7d78d7c83ddd17c4c534d917))

- Move intensity threshold filtering to reader level
  ([#71](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/71),
  [`ab1dfe4`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/ab1dfe4f9fc9f81c7d78d7c83ddd17c4c534d917))

- Strategy pattern for instrument detection and continuous mode optimization
  ([#72](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/72),
  [`74c1f29`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/74c1f29b8d4dbdfa8a220ca800d5f7ed98fbc181))


## v1.12.1 (2026-01-23)

### Bug Fixes

- Lower PCS threshold from 50 GB to 30 GB for memory efficiency
  ([`74f6700`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/74f6700be667408b07cef2e9f3b78bcb3647a5a8))


## v1.12.0 (2026-01-23)

### Bug Fixes

- Support datasets with >2.1 billion non-zeros in streaming converter
  ([`f111ed8`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/f111ed835dd4d34ee0d4c7e44b4a8412e687f99d))


## v1.11.1 (2026-01-23)

### Bug Fixes

- Handle ResamplingConfig dataclass in streaming converter
  ([`abb9d06`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/abb9d061a683be4355bb879a99fbb4dabd0a5d46))


## v1.11.0 (2026-01-23)

### Features

- Add streaming parameter to convert_msi API
  ([#70](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/pull/70),
  [`37c7d2d`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/37c7d2df313eea358873312aa51c6fe6fc61b930))


## v1.10.0 (2026-01-23)

### Bug Fixes

- Correct release workflow YAML syntax and job dependencies
  ([`5ccf1ec`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/5ccf1ec66c3f28a4f7bdc068c738ccd08a03df01))

### Code Style

- Apply black formatting to streaming converter
  ([`1432d27`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/1432d277270dc936289d215980b78e7014db1dc4))

### Features

- Add streaming converter for memory-efficient large dataset conversion
  ([`55f7d42`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/55f7d4294d133a10f0281afe2bba49e3ee93a880))

- Add streaming converter for memory-efficient large dataset conversion
  ([`fd7acf5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/fd7acf5a2aedcf31855f7dc7d39f962dc5e2e4a1))

- Implement no-cache CSC streaming for memory-efficient large dataset conversion
  ([`4bac2d1`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/4bac2d1560349bee46dbc0076fddf5a9384a3391))

### Refactoring

- Reduce _get_mass_range_complete complexity from 13 to ~5
  ([`94bcc95`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/94bcc952b90eafd95afd7fd5d67e7c9bacdeba1b))

- Reduce _stream_build_coo complexity from 16 to ~7
  ([`69b5ba9`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/69b5ba92806d7b94c4fb3644039f2e641c520eac))

- Remove dead code (zero_copy parameter and _convert_with_scipy)
  ([`d6affd5`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/d6affd520d186fc1e4f21ae97fa20a1ca10d4f68))

- Streamline streaming converter code
  ([`29a834f`](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/commit/29a834fa9b1cfaccfbc14ba08168a8724e8faf5c))


## v1.9.0 (2025-12-15)

- Initial Release
