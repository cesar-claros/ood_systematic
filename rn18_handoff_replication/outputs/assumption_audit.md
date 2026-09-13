# Assumption audit (matched comparisons)

```
{
 "rule": {
  "R1_level_err": 0.05,
  "R1_min_sources": 3,
  "R2_delta_err": 0.05,
  "R2_delta_sign": 0.1,
  "R3_delta_err": 0.05,
  "R3_delta_sign": 0.1,
  "R4_sign_agreement": 0.75
 },
 "R2_detail": {
  "train|G0": {
   "d_err": 0.01985824278626447,
   "d_sign": -0.043165467625899345
  },
  "train|G1": {
   "d_err": 0.02783774047082936,
   "d_sign": -0.06834532374100721
  },
  "test|G0": {
   "d_err": 0.02311425013214298,
   "d_sign": 0.017985611510791366
  },
  "test|G1": {
   "d_err": 0.0455805605770168,
   "d_sign": -0.03956834532374098
  },
  "iso_train|G0": {
   "d_err": 0.02446866931607311,
   "d_sign": -0.014388489208633004
  },
  "iso_train|G1": {
   "d_err": 0.03138182333846805,
   "d_sign": -0.03597122302158273
  },
  "iso_test|G0": {
   "d_err": 0.030532357495515583,
   "d_sign": 0.010791366906474864
  },
  "iso_test|G1": {
   "d_err": 0.05310990803625537,
   "d_sign": -0.014388489208633115
  }
 },
 "R3_detail": {
  "train|trainvsiso_train|G0": {
   "d_err": 0.0012312402816560541,
   "d_sign": -0.039568345323741094
  },
  "train|testvsiso_test|G0": {
   "d_err": 0.006415902116787789,
   "d_sign": -0.007194244604316613
  },
  "train|trainvsiso_train|G1": {
   "d_err": 0.00396355646649664,
   "d_sign": -0.043165467625899234
  },
  "train|testvsiso_test|G1": {
   "d_err": 0.012876620716949602,
   "d_sign": -0.05035971223021579
  },
  "test|trainvsiso_train|G0": {
   "d_err": 0.0047169487503508495,
   "d_sign": -0.017985611510791366
  },
  "test|testvsiso_test|G0": {
   "d_err": 0.024974199462640165,
   "d_sign": -0.010791366906474864
  },
  "test|trainvsiso_train|G1": {
   "d_err": 0.010942204363935532,
   "d_sign": -0.003597122302158251
  },
  "test|testvsiso_test|G1": {
   "d_err": 0.03092371546279142,
   "d_sign": -0.02877697841726623
  },
  "etf|trainvsiso_train|G0": {
   "d_err": 0.005841666811464724,
   "d_sign": -0.010791366906474753
  },
  "etf|testvsiso_test|G0": {
   "d_err": 0.013834009480160378,
   "d_sign": -0.014388489208633115
  },
  "etf|trainvsiso_train|G1": {
   "d_err": 0.007507639334135341,
   "d_sign": -0.010791366906474753
  },
  "etf|testvsiso_test|G1": {
   "d_err": 0.02040596817618817,
   "d_sign": -0.025179856115107924
  }
 },
 "origin_check_bprime": {
  "cifar10": {
   "dE": -0.0004310859875245876,
   "dC": 0.0003760811957446153
  },
  "cifar100": {
   "dE": 0.00010310982664424317,
   "dC": 0.0040780382851759445
  },
  "supercifar100": {
   "dE": 1.9027718475894684e-05,
   "dC": 0.00500622753586083
  },
  "tinyimagenet": {
   "dE": 0.0001520194790579632,
   "dC": 0.01827820485288445
  }
 },
 "verdict": {
  "R1_feature_statistics_generalize": {
   "test|test|G0": false,
   "test|test|G1": false
  },
  "R2_etf_mean_restriction_material": {
   "train|G0": false,
   "train|G1": false,
   "test|G0": false,
   "test|G1": false,
   "iso_train|G0": false,
   "iso_train|G1": false,
   "iso_test|G0": false,
   "iso_test|G1": true
  },
  "R3_covariance_shape_matters": {
   "train|trainvsiso_train|G0": false,
   "train|testvsiso_test|G0": false,
   "train|trainvsiso_train|G1": false,
   "train|testvsiso_test|G1": false,
   "test|trainvsiso_train|G0": false,
   "test|testvsiso_test|G0": false,
   "test|trainvsiso_train|G1": false,
   "test|testvsiso_test|G1": false,
   "etf|trainvsiso_train|G0": false,
   "etf|testvsiso_test|G0": false,
   "etf|trainvsiso_train|G1": false,
   "etf|testvsiso_test|G1": false
  },
  "R4_usable_gap_prediction": [],
  "best_level_variant": "test|test|G0"
 },
 "by_variant_all": {
  "etf|iso_test|G0": {
   "n": 384,
   "bias_E": 0.15862776692050554,
   "bias_C": 0.026322823954086024,
   "mae_E": 0.2552110487948364,
   "mae_C": 0.2531522598225639,
   "gap_mae": 0.13803116458274398,
   "sign_agree_material": 0.5683453237410072,
   "n_material": 278
  },
  "etf|iso_test|G1": {
   "n": 384,
   "bias_E": 0.1345391696228371,
   "bias_C": -0.1418233680179114,
   "mae_E": 0.2508098500523805,
   "mae_C": 0.2310186919506608,
   "gap_mae": 0.27661893054352454,
   "sign_agree_material": 0.420863309352518,
   "n_material": 278
  },
  "etf|iso_train|G0": {
   "n": 384,
   "bias_E": 0.15864306184488394,
   "bias_C": 0.11531017497674677,
   "mae_E": 0.2558894301186508,
   "mae_C": 0.2941174574517635,
   "gap_mae": 0.07263317828822319,
   "sign_agree_material": 0.564748201438849,
   "n_material": 278
  },
  "etf|iso_train|G1": {
   "n": 384,
   "bias_E": 0.13482028716889877,
   "bias_C": 0.042334145809174106,
   "mae_E": 0.25167875594683875,
   "mae_C": 0.2954352697949533,
   "gap_mae": 0.11256879739339083,
   "sign_agree_material": 0.564748201438849,
   "n_material": 278
  },
  "etf|test|G0": {
   "n": 384,
   "bias_E": 0.15956087173608793,
   "bias_C": 0.025784845593035722,
   "mae_E": 0.23523886907112987,
   "mae_C": 0.24545642058594966,
   "gap_mae": 0.14077811327404272,
   "sign_agree_material": 0.5827338129496403,
   "n_material": 278
  },
  "etf|test|G1": {
   "n": 384,
   "bias_E": 0.12518539862422004,
   "bias_C": -0.13186940003935446,
   "mae_E": 0.22373748940179206,
   "mae_C": 0.21727911624887286,
   "gap_mae": 0.2572247837578719,
   "sign_agree_material": 0.4460431654676259,
   "n_material": 278
  },
  "etf|train|G0": {
   "n": 384,
   "bias_E": 0.160446659963209,
   "bias_C": 0.11635693113501917,
   "mae_E": 0.24696636989171009,
   "mae_C": 0.29135718405577476,
   "gap_mae": 0.0758521869023983,
   "sign_agree_material": 0.5755395683453237,
   "n_material": 278
  },
  "etf|train|G1": {
   "n": 384,
   "bias_E": 0.13216722234177447,
   "bias_C": 0.044614899535934636,
   "mae_E": 0.24081178441096804,
   "mae_C": 0.2912869626625534,
   "gap_mae": 0.11150405758988828,
   "sign_agree_material": 0.5755395683453237,
   "n_material": 278
  },
  "test|iso_test|G0": {
   "n": 384,
   "bias_E": -0.008238225563487836,
   "bias_C": -0.022391553776998807,
   "mae_E": 0.12179414342714028,
   "mae_C": 0.169926332798061,
   "gap_mae": 0.07166222348782554,
   "sign_agree_material": 0.6258992805755396,
   "n_material": 278
  },
  "test|iso_test|G1": {
   "n": 384,
   "bias_E": -0.09146399275374556,
   "bias_C": -0.21321729247275933,
   "mae_E": 0.13620672534506376,
   "mae_C": 0.23041361736862076,
   "gap_mae": 0.12263654507510822,
   "sign_agree_material": 0.539568345323741,
   "n_material": 278
  },
  "test|iso_train|G0": {
   "n": 384,
   "bias_E": -0.00979226741604152,
   "bias_C": 0.1384911052311068,
   "mae_E": 0.12307456568760132,
   "mae_C": 0.16265508983110813,
   "gap_mae": 0.15776766734597444,
   "sign_agree_material": 0.6151079136690647,
   "n_material": 278
  },
  "test|iso_train|G1": {
   "n": 384,
   "bias_E": -0.09263613361529653,
   "bias_C": 0.040041411424637345,
   "mae_E": 0.13777004746689234,
   "mae_C": 0.13759992803925444,
   "gap_mae": 0.15369302503630167,
   "sign_agree_material": 0.6294964028776978,
   "n_material": 278
  },
  "test|test|G0": {
   "n": 384,
   "bias_E": 0.027630409837641885,
   "bias_C": -0.0010251430745318406,
   "mae_E": 0.09109118219211827,
   "mae_C": 0.15068089510780266,
   "gap_mae": 0.07474768547982884,
   "sign_agree_material": 0.6366906474820144,
   "n_material": 278
  },
  "test|test|G1": {
   "n": 384,
   "bias_E": -0.06375764752092107,
   "bias_C": -0.18890194426729232,
   "mae_E": 0.09990790517622465,
   "mae_C": 0.20486500661187698,
   "gap_mae": 0.1297854042454294,
   "sign_agree_material": 0.5683453237410072,
   "n_material": 278
  },
  "test|train|G0": {
   "n": 384,
   "bias_E": -0.0021084948843930564,
   "bias_C": 0.14438243806854933,
   "mae_E": 0.11455835580501043,
   "mae_C": 0.1617374022129973,
   "gap_mae": 0.1545585620329131,
   "sign_agree_material": 0.6330935251798561,
   "n_material": 278
  },
  "test|train|G1": {
   "n": 384,
   "bias_E": -0.08751998240413411,
   "bias_C": 0.04686157126472445,
   "mae_E": 0.12834841380850218,
   "mae_C": 0.12513715296977354,
   "gap_mae": 0.15243092766079294,
   "sign_agree_material": 0.6330935251798561,
   "n_material": 278
  },
  "train|iso_test|G0": {
   "n": 384,
   "bias_E": 0.2422187548661019,
   "bias_C": 0.1906390844583119,
   "mae_E": 0.24275780484255938,
   "mae_C": 0.20454078878380977,
   "gap_mae": 0.06926413447329095,
   "sign_agree_material": 0.5575539568345323,
   "n_material": 278
  },
  "train|iso_test|G1": {
   "n": 384,
   "bias_E": 0.23614743462788915,
   "bias_C": 0.09355452017730526,
   "mae_E": 0.23664548480437983,
   "mae_C": 0.1389632411261507,
   "gap_mae": 0.14727396756199404,
   "sign_agree_material": 0.4352517985611511,
   "n_material": 278
  },
  "train|iso_train|G0": {
   "n": 384,
   "bias_E": 0.2424315540650274,
   "bias_C": 0.25521769350266504,
   "mae_E": 0.2429954421661972,
   "mae_C": 0.2580741067720709,
   "gap_mae": 0.034893343265216405,
   "sign_agree_material": 0.579136690647482,
   "n_material": 278
  },
  "train|iso_train|G1": {
   "n": 384,
   "bias_E": 0.23662012479293204,
   "bias_C": 0.23794882676746498,
   "mae_E": 0.23715423590368814,
   "mae_C": 0.24719614316116792,
   "gap_mae": 0.039781015333984315,
   "sign_agree_material": 0.6007194244604317,
   "n_material": 278
  },
  "train|test|G0": {
   "n": 384,
   "bias_E": 0.2348036094788497,
   "bias_C": 0.19017665305978113,
   "mae_E": 0.2348036094788497,
   "mae_C": 0.19966317991394386,
   "gap_mae": 0.06228897404921058,
   "sign_agree_material": 0.564748201438849,
   "n_material": 278
  },
  "train|test|G1": {
   "n": 384,
   "bias_E": 0.21975163828917202,
   "bias_C": 0.0966486676955824,
   "mae_E": 0.21975163828917202,
   "mae_C": 0.1301038462074593,
   "gap_mae": 0.12637183046235445,
   "sign_agree_material": 0.4856115107913669,
   "n_material": 278
  },
  "train|train|G0": {
   "n": 384,
   "bias_E": 0.24060622514871613,
   "bias_C": 0.25662995242075176,
   "mae_E": 0.24060622514871613,
   "mae_C": 0.25800084322623984,
   "gap_mae": 0.03336516189068216,
   "sign_agree_material": 0.6187050359712231,
   "n_material": 278
  },
  "train|train|G1": {
   "n": 384,
   "bias_E": 0.23144428263443564,
   "bias_C": 0.2390857601587579,
   "mae_E": 0.23144428263443564,
   "mae_C": 0.2449789834974271,
   "gap_mae": 0.03853650185001545,
   "sign_agree_material": 0.6438848920863309,
   "n_material": 278
  },
  "train|train|G1|centered_bprime": {
   "n": 384,
   "bias_E": 0.23141165710079528,
   "bias_C": 0.24584035998082213,
   "mae_E": 0.23141165710079528,
   "mae_C": 0.2510090571112249,
   "gap_mae": 0.04180063802569242,
   "sign_agree_material": 0.6187050359712231,
   "n_material": 278
  }
 }
}
```
