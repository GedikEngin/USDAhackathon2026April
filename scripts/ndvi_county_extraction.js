// =============================================================================
// ndvi_county_extraction.js
// =============================================================================
// USDA Hackathon 2026 April — Phase A.2
//
// PURPOSE
//   Pull MODIS NDVI features per (county, year) for 5 corn-belt states,
//   masked to corn-only pixels via USDA CDL. All aggregation is done
//   server-side in Earth Engine. Output is one CSV per year, already at
//   county-year granularity, written to Google Drive folder
//   `EarthEngineExports/`.
//
// HOW TO RUN
//   1. Paste this whole file into the Earth Engine Code Editor
//      (https://code.earthengine.google.com/).
//   2. Run. The script enqueues one Export.table.toDrive task per year
//      in the [START_YEAR, END_YEAR] range (inclusive on both ends).
//   3. Open the Tasks tab and click Run on each (or run them all).
//   4. Wait. Each year takes ~5–15 minutes depending on EE queue depth.
//   5. Download `EarthEngineExports/corn_ndvi_5states_<year>.csv` from
//      Drive into `phase2/data/ndvi/` in the repo.
//
// HOW TO BACKFILL OR EXTEND
//   Change START_YEAR / END_YEAR below and re-run. CDL 2008 is the
//   earliest corn-class-stable year for many states — for years
//   before 2008, CDL coverage may be partial (some states missing).
//   CDL 2024 publishes Jan/Feb 2025; if 2024 fails with a "no image
//   matching filter" error, set END_YEAR = 2023.
//
// OUTPUT SCHEMA (one row per county-year, one file per year)
//   GEOID            5-digit county FIPS (string, zero-padded)
//   NAME             County name
//   STATEFP          2-digit state FIPS
//   year             Year (int)
//   ndvi_peak        Max NDVI during growing season (DOY 121–273)
//   ndvi_gs_mean     Mean NDVI during growing season
//   ndvi_gs_integral Sum of NDVI values during growing season
//   ndvi_silking_mean Mean NDVI during silking window (DOY 196–227)
//   ndvi_veg_mean    Mean NDVI during vegetative phase (DOY 152–195)
//
//   NOTE: NDVI values in the CSV are already scaled (multiplied by
//   0.0001) — they are floats in roughly [-0.2, 1.0]. No further
//   scaling needed downstream.
//
// BUG FIXES BAKED IN (do not remove)
//   1. CDL is loaded as `ee.ImageCollection('USDA/NASS/CDL').filter(...)`
//      and NOT via string-concat like `'USDA/NASS/CDL/' + year`. Server-side
//      `ee.Number` cannot be JS-concatenated into asset paths; the wrong
//      pattern produces invalid asset paths that fail at evaluation time.
//   2. The map demo image (commented out by default below) is wrapped with
//      `ee.Image(...).set('system:description', 'demo')` to reset
//      accumulated metadata that otherwise causes Map.addLayer description
//      errors after multiple runs in the same editor session.
//
// DEPENDENCIES (all are public EE assets, no auth needed beyond your account)
//   - MODIS/061/MOD13Q1            16-day NDVI composites, 250m
//   - USDA/NASS/CDL                Annual cropland data layer, 30m
//   - TIGER/2018/Counties          County polygons (used for reduction)
// =============================================================================

// ---------- config ----------------------------------------------------------

var START_YEAR = 2004;   // inclusive. CDL pre-2008 has partial state coverage.
var END_YEAR   = 2024;   // inclusive. If CDL <year> not yet published, drop by 1.

// 5 target states by FIPS code: IA=19, CO=08, WI=55, MO=29, NE=31.
var STATE_FIPS = ['19', '08', '55', '29', '31'];

// Growing-season windows (DOY, inclusive). Tuned for corn in the upper Midwest.
var GS_DOY_START   = 121;  // ~May 1
var GS_DOY_END     = 273;  // ~Sep 30
var VEG_DOY_START  = 152;  // ~Jun 1   — vegetative phase
var VEG_DOY_END    = 195;  // ~Jul 14
var SILK_DOY_START = 196;  // ~Jul 15  — silking window
var SILK_DOY_END   = 227;  // ~Aug 15

// MODIS scale factor — NDVI band stored as int16, multiply by 0.0001.
var NDVI_SCALE = 0.0001;

// CDL corn class value.
var CDL_CORN = 1;

// Output folder in Drive (created automatically if missing).
var DRIVE_FOLDER = 'EarthEngineExports';

// Reduction scale (m) — match MODIS native res.
var REDUCE_SCALE = 250;
var TILE_SCALE   = 4;       // bump to 8 or 16 if "computation timed out".

// ---------- counties --------------------------------------------------------

// TIGER/Line 2018 county polygons, filtered to the 5 target states.
// Stable across the 2004–2024 window (no county boundary changes that affect us).
var counties = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.inList('STATEFP', STATE_FIPS))
  .map(function(f) {
    // Keep only the columns we need in the output, drop the rest.
    return f.select(['GEOID', 'NAME', 'STATEFP']);
  });

// ---------- per-year feature builder ----------------------------------------

// Build one FeatureCollection (one row per county) for a given year.
// Uses CDL <year> to mask MODIS NDVI to corn pixels, then reduces over
// county polygons for each of the 5 NDVI summary statistics.
function ndviForYear(year) {
  year = ee.Number(year);

  // ---- corn mask from CDL --------------------------------------------------
  // Load CDL via ImageCollection().filter() — NOT via string concat.
  // String-concat fails because `year` is server-side ee.Number.
  var cdlStart = ee.Date.fromYMD(year, 1, 1);
  var cdlEnd   = ee.Date.fromYMD(year, 12, 31);
  var cdl = ee.ImageCollection('USDA/NASS/CDL')
    .filterDate(cdlStart, cdlEnd)
    .first();
  var cornMask = ee.Image(cdl).select('cropland').eq(CDL_CORN);

  // ---- MODIS NDVI for the growing season ----------------------------------
  var gsStart = ee.Date.fromYMD(year, 1, 1).advance(GS_DOY_START - 1, 'day');
  var gsEnd   = ee.Date.fromYMD(year, 1, 1).advance(GS_DOY_END,     'day');
  var modis = ee.ImageCollection('MODIS/061/MOD13Q1')
    .filterDate(gsStart, gsEnd)
    .select('NDVI')
    .map(function(img) {
      // Apply scale factor and re-attach DOY for downstream filters.
      var doy = ee.Number.parse(img.date().format('DDD'));
      return img.multiply(NDVI_SCALE)
        .updateMask(cornMask)
        .set('doy', doy)
        .copyProperties(img, ['system:time_start']);
    });

  // ---- summary images ------------------------------------------------------
  // Whole growing season.
  var gsPeak     = modis.max().rename('ndvi_peak');
  var gsMean     = modis.mean().rename('ndvi_gs_mean');
  var gsIntegral = modis.sum().rename('ndvi_gs_integral');

  // Silking window (DOY 196–227).
  var silkColl = modis.filter(ee.Filter.rangeContains('doy', SILK_DOY_START, SILK_DOY_END));
  var silkMean = silkColl.mean().rename('ndvi_silking_mean');

  // Vegetative window (DOY 152–195).
  var vegColl = modis.filter(ee.Filter.rangeContains('doy', VEG_DOY_START, VEG_DOY_END));
  var vegMean = vegColl.mean().rename('ndvi_veg_mean');

  // Stack into one 5-band image, all corn-masked.
  var stack = gsPeak
    .addBands(gsMean)
    .addBands(gsIntegral)
    .addBands(silkMean)
    .addBands(vegMean);

  // ---- reduce over county polygons ----------------------------------------
  // One feature per county. Mean reducer = mean of the per-pixel summary
  // images across all corn pixels in the county. Counties with zero corn
  // pixels (CDL == 1) will come out null — handle downstream.
  var reduced = stack.reduceRegions({
    collection: counties,
    reducer:    ee.Reducer.mean(),
    scale:      REDUCE_SCALE,
    tileScale:  TILE_SCALE,
  });

  // Tag every feature with the year column.
  reduced = reduced.map(function(f) { return f.set('year', year); });

  return reduced;
}

// ---------- export loop -----------------------------------------------------

// One Export task per year. Years export independently — if one fails
// (e.g., CDL not yet published), the rest still complete.
for (var y = START_YEAR; y <= END_YEAR; y++) {
  var fc = ndviForYear(y);

  Export.table.toDrive({
    collection:    fc,
    description:   'corn_ndvi_5states_' + y,                // task name in EE
    folder:        DRIVE_FOLDER,
    fileNamePrefix:'corn_ndvi_5states_' + y,                // CSV filename (no .csv)
    fileFormat:    'CSV',
    selectors: [
      'GEOID', 'NAME', 'STATEFP', 'year',
      'ndvi_peak', 'ndvi_gs_mean', 'ndvi_gs_integral',
      'ndvi_silking_mean', 'ndvi_veg_mean'
    ],
  });
}

// ---------- optional: visual sanity check (commented) -----------------------
// Uncomment one of these blocks to QC a single year on the EE map.
// Wrap any image you addLayer() with `ee.Image(...).set('system:description', 'demo')`
// — otherwise Map.addLayer can throw description errors after several runs.
//
// var qcYear = 2020;
// var qcFc = ndviForYear(qcYear);
// print('QC ' + qcYear + ' first 5 features:', qcFc.limit(5));
// print('QC ' + qcYear + ' size:', qcFc.size());
//
// var qcImg = ee.Image(
//   ee.ImageCollection('MODIS/061/MOD13Q1')
//     .filterDate(qcYear + '-07-15', qcYear + '-08-15')
//     .select('NDVI')
//     .mean()
//     .multiply(NDVI_SCALE)
// ).set('system:description', 'demo');
// Map.centerObject(counties, 5);
// Map.addLayer(counties.style({color: 'white', fillColor: '00000000'}), {}, 'counties');
// Map.addLayer(qcImg, {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI ' + qcYear);