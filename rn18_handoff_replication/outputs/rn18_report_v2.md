# rn18_report_v2

```
{
 "reader": "rn18_analysis_v2.py",
 "validation": {
  "git_head_at_run": "be7e3ffc28037965bd0a27a691abef5c5da94be7",
  "checks": {
   "reader_of_record_discrepancy": {
    "frozen": "038cf0446f3e576fdccfc9021a685be76e7aba933715ba148a39fa58e28066a1",
    "reader_of_record": "c50c47fa8ff2b23069396e58dddf7e2a505eca354b32c9ceefc63981465b9203",
    "commit": "c7d1b98",
    "attribution": "frozen hash 038cf044... (commit 6ae5b4b) vs reader of record c50c47fa... (commit c7d1b98): the only difference is the JSON serialization of the tertile composition keys; no numerical path changed"
   },
   "freeze_hashes": {
    "n_checked": 28,
    "mismatches": []
   },
   "reader_v2_sha256": "6f993c3a1cba4d21caee4469b2694ad5e668b74e3468f6a5b13b9a92b0550db6",
   "inventory:fourshift_rn18": {
    "n": 96,
    "extra": [],
    "missing": [],
    "changed": [],
    "other_files": [],
    "sidecars_not_in_freeze_inventory": {
     "n": 96,
     "rule": "recorded, not inventoried by the freeze",
     "files": [
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do0_run1_rew0.npz",
       "bytes": 10340583,
       "sha256": "d433742fd3c44178562fdc3adb88e7d2fd78a0ea1e9fe882485680cb236c9f9d"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do0_run2_rew0.npz",
       "bytes": 10315947,
       "sha256": "e2319eca4e705e84535694d7d6b1634a35a4ae0fc2b2e6c8780a5378b953b356"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do0_run3_rew0.npz",
       "bytes": 10335320,
       "sha256": "3d4bca6997535b9f68b38b62a8eba5f0f8aead0bb6bd43b6a9f4029f6f9e5493"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do0_run4_rew0.npz",
       "bytes": 10347745,
       "sha256": "50e2e2c81fda4a46651cba03a1064229bef4b10b6c4ffe3c7abbfbd6fd43a0ad"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do0_run5_rew0.npz",
       "bytes": 10329485,
       "sha256": "7901cd8eff66fc85e8a1053fdb795ddf0d280017b75078d7418ea16af62ffe75"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do1_run1_rew0.npz",
       "bytes": 10259155,
       "sha256": "4dfbff74f3ada0124f76c63469c58c24445e93ec773be9e721ffa5c5238e1365"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do1_run2_rew0.npz",
       "bytes": 10271958,
       "sha256": "6efecaeca28f6c52b507eb27d55fe1c4974bb77111cbd0415868a0faf88ae0bf"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do1_run3_rew0.npz",
       "bytes": 10280326,
       "sha256": "f28c55168b43a84fe9295a4b22e91567c9c7abc708017e41f12dd65607977c0f"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do1_run4_rew0.npz",
       "bytes": 10269888,
       "sha256": "d750ab2d04a009d2035e23e073511974d6a8d238a4087a539610c78777429116"
      },
      {
       "name": "cifar100_paper_sweep__ce_bbresnet18_do1_run5_rew0.npz",
       "bytes": 10274505,
       "sha256": "a9914ad261893aeb37e947d66fa01b49bda6df4f0308372b384895cdb02c2362"
      },
      {
       "name": "cifar100_paper_sweep__confidnet_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 10339547,
       "sha256": "06c43d24da5335bc976d18ffb59f3094dee042ae55dce1e854d7de348cf6e10d"
      },
      {
       "name": "cifar100_paper_sweep__confidnet_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 5246649,
       "sha256": "6b7d5157cd4af9b3a8881807d57559962346b33a120578f62048c44e98eb411c"
      },
      {
       "name": "cifar100_paper_sweep__devries_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 10322424,
       "sha256": "011c0fd6e828c33a22527086abf317227b3e1297652752110d4a9533c7af86c2"
      },
      {
       "name": "cifar100_paper_sweep__devries_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 6112893,
       "sha256": "9331b257101903c493f804b4b18f391cde4ae3c8a865cc19ec8549a7cd3950bc"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do0_run1_rew10.npz",
       "bytes": 10346263,
       "sha256": "86be53b3caf3530514850e30b57d88b8cf94891e4b2d6b055ef59132635eb787"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do0_run1_rew12.npz",
       "bytes": 10340663,
       "sha256": "b944886858485a0048283e024c197556a62260ef5fd1324247592cebe02ddec5"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do0_run1_rew15.npz",
       "bytes": 10331137,
       "sha256": "a7ea60ffba861f2aef63e785ed29f96f0fc655de937e9de3465099e45b35bb28"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do0_run1_rew20.npz",
       "bytes": 10352677,
       "sha256": "5b9828ed0fed55cf3c9f3828597e4759157945dc69f6b1605ec0389cb95ae73a"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do0_run1_rew6.npz",
       "bytes": 10339177,
       "sha256": "a0140cc950beffb63c2925ba63f8be5fb8b8d6d1a969fd9e7d3e0cb8a483785e"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do1_run1_rew10.npz",
       "bytes": 5383266,
       "sha256": "ec3df47c4c33f127e6ad2b032ee7ec85efa133c4c4ecfccd36ae70fa9a606b02"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do1_run1_rew12.npz",
       "bytes": 5563511,
       "sha256": "62ea4bfa1d4c6682fabd311a351b7af143a06644fcf284431b61b687a97b30f4"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do1_run1_rew15.npz",
       "bytes": 5899101,
       "sha256": "221f08da218583ff7a325bbdd57fce8caf0adf7fc8c5ae10cef298d95355ecec"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do1_run1_rew20.npz",
       "bytes": 5407552,
       "sha256": "0e6621cee54eb90437951dd502e194fbe619b4556e01c7825f2540e6a740fbde"
      },
      {
       "name": "cifar100_paper_sweep__dg_bbresnet18_do1_run1_rew6.npz",
       "bytes": 5829957,
       "sha256": "7c16d8b12814d70bcd4dea7a8c66179685d1f3071080eacf60cb9258667d2855"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do0_run1_rew0.npz",
       "bytes": 9106489,
       "sha256": "7ef873cd6ceabb07d070964c309899a1e040a746aa044987fe1addecd8d2d892"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do0_run2_rew0.npz",
       "bytes": 9062224,
       "sha256": "57b6c81a566edbab386943ae3864ccda44fb5b8eca8225cef93d7adb984714e6"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do0_run3_rew0.npz",
       "bytes": 9080238,
       "sha256": "768e43050830dd05bd2f14a867a84b05a557a90844978d1408ddc453b64cdf45"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do0_run4_rew0.npz",
       "bytes": 9108723,
       "sha256": "a3f3f3a43d6b0617efb504ad70247d67ba70960967de5d8eb944f0963972bba2"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do0_run5_rew0.npz",
       "bytes": 9087789,
       "sha256": "78ee4de0804cb4596b045b4d56b0f2f2af418a3e134d8feb96e2c8ac25059422"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do1_run1_rew0.npz",
       "bytes": 9001406,
       "sha256": "630a63c4fb7e3e9527b94172ae1b059c5a8dde06902a2fd76c6e8f6bfb4ec491"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do1_run2_rew0.npz",
       "bytes": 9010158,
       "sha256": "4942982c645dcd97606b73f87ccf5b91d60dc69a80acbbea5b2248c6779ddd50"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do1_run3_rew0.npz",
       "bytes": 9001594,
       "sha256": "f8b6d9c222ef837283bb55232a3c5a47085032773fcf2181b13eb8a2f460f7ea"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do1_run4_rew0.npz",
       "bytes": 8997671,
       "sha256": "bedbbaf59b938555f7d3fdffe6c83daf45975d2cbde96dc8850cd6520858f0b9"
      },
      {
       "name": "cifar10_paper_sweep__ce_bbresnet18_do1_run5_rew0.npz",
       "bytes": 9004686,
       "sha256": "e96d04f3d8cdaa25016e075da4e9849b7251d55b570cacc736906aac2f85dd8a"
      },
      {
       "name": "cifar10_paper_sweep__confidnet_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 9058560,
       "sha256": "dede860d9e67ed0d1d10884dcbb84648382ea3556bc0a3e0efbae497ab34ed49"
      },
      {
       "name": "cifar10_paper_sweep__confidnet_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 4722540,
       "sha256": "fcc322ee522718d4cc2990e18377d355c7ea9dfd268e30cb507fdf71c7a25ce4"
      },
      {
       "name": "cifar10_paper_sweep__devries_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 9068628,
       "sha256": "9c84c3e3184d629aee07f8e3169db684f8d4ac6120dcaa35b2510b7548ac8ab9"
      },
      {
       "name": "cifar10_paper_sweep__devries_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 4539434,
       "sha256": "9120bdbd06dc775f1d2bac537d6d0d6208f78efce5f5d1d35cdc3a30c4643c88"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew10.npz",
       "bytes": 9073666,
       "sha256": "bbac2dfd3ecc16cd6da22903173c60945a4ecaf3290ad44988704d4f16bcdcd2"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 9091249,
       "sha256": "6714a9e2baaab4a21d73056d04f31e9a8ae1e04ad3976c818915d8d75f443f3e"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew3.npz",
       "bytes": 9086375,
       "sha256": "92bf6af3f47201a7f25aaf4ee5691efdf17a003272086215a2cc71b6150e76bd"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do0_run1_rew6.npz",
       "bytes": 9068273,
       "sha256": "a7f0467a7fa9b2b1257f1dc5db798e186217eae54b56066ecab48b15023fbebe"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do1_run1_rew10.npz",
       "bytes": 5008394,
       "sha256": "ea88d75d48b7fde4c3ceaed84906252120c1481a055393abb427bdab230fdcb2"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 5934303,
       "sha256": "49b281f4b37dd402e9606a9870d8ec827c1e8225a78cd175647866c7501c2c04"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do1_run1_rew3.npz",
       "bytes": 4810510,
       "sha256": "4d47859f163abce9aaa2f22cdb9e7d2346777c059beaaf853157b47e2cfac238"
      },
      {
       "name": "cifar10_paper_sweep__dg_bbresnet18_do1_run1_rew6.npz",
       "bytes": 4828517,
       "sha256": "20859ac6d52925a64a26a579146af2884cb8ca1c74a852a19eeb2dd2e789cdb8"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do0_run1_rew0.npz",
       "bytes": 9216474,
       "sha256": "a02c09228dbdbf72b0af9077a0313e433d7d403d0853b13e8ef07e0672b501c8"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do0_run2_rew0.npz",
       "bytes": 9227270,
       "sha256": "1e3baacda195e57a217356908d5e48e989589f6ccf292895d72b85e940e7a00f"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do0_run3_rew0.npz",
       "bytes": 9209855,
       "sha256": "597621e87a9011ea77214693dd00a2221f1239923813e078b88b283627050dab"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do0_run4_rew0.npz",
       "bytes": 9209614,
       "sha256": "b4b6319cdc32541081e82022e0d4b060b430b76255f02dae9a5a7a5fbc356277"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do0_run5_rew0.npz",
       "bytes": 9211236,
       "sha256": "a917e3c6551201a7fa6de5b6070947fdb3d09a4962778123bafd2e7fddeef0be"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do1_run1_rew0.npz",
       "bytes": 9184293,
       "sha256": "d0e6bec1c9194db2327b8c539ae36a6ab3562944a00332a57017265dd50337a9"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do1_run2_rew0.npz",
       "bytes": 9197280,
       "sha256": "a653a141a705ae63969894e58e64611daea2500035ccbb1040fc9ac5881c4772"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do1_run3_rew0.npz",
       "bytes": 9190602,
       "sha256": "84d6acc59c6435cbf7e339d13a04160481bd1770f70e823f5983ebe689f31870"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do1_run4_rew0.npz",
       "bytes": 9186912,
       "sha256": "b1c8c05e964126c5caaedf927b2a0c0325adf052b1490ba85ffcd7785575cb75"
      },
      {
       "name": "supercifar_paper_sweep__ce_bbresnet18_do1_run5_rew0.npz",
       "bytes": 9194359,
       "sha256": "dafe8868a772eb5da306d1497c3670028af3ccf5a277accc91889393b55bf85b"
      },
      {
       "name": "supercifar_paper_sweep__confidnet_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 9243163,
       "sha256": "53a3d28f75794e16a9dd9e3c8f8794563db1dc046d01dab6a444954d39d10e13"
      },
      {
       "name": "supercifar_paper_sweep__confidnet_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 9209619,
       "sha256": "0a8aa149d1d06d2fbb1bc66765dc707c85c0c2cc00e283ebdb8f3310b3b0b3eb"
      },
      {
       "name": "supercifar_paper_sweep__devries_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 9243772,
       "sha256": "2e1b8cf5e0ec127b22007e04d92c419943e235935f900412df15f28ee3fea40d"
      },
      {
       "name": "supercifar_paper_sweep__devries_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 9275961,
       "sha256": "be585b0a44045374d7aebf151c3506e6c4256bfb6c42576c0ea753a45467644c"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do0_run1_rew10.npz",
       "bytes": 9264228,
       "sha256": "dd45ef9814154c9205b914990641f007d31a97de090ae307e92fa92e9fa89ede"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do0_run1_rew12.npz",
       "bytes": 9242100,
       "sha256": "16718d2c036c6588ccb3ffc7fd1731064e0653b324d6536164e51ba1b69dca70"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do0_run1_rew15.npz",
       "bytes": 9239982,
       "sha256": "b5de7baec2d564709fa29b5d02fd07dd23dc2e32a43d3eef0bb2f0a70a074f68"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 9263273,
       "sha256": "d60ad46e3c706a42cd628ffd15563c5e743ae959ca6c26d6f0d5714b76f4bcc2"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do0_run1_rew20.npz",
       "bytes": 9233381,
       "sha256": "aacbfc28bde3d64137ad1d2d8715bb71eb3b153f0ce9eb66aabc48a475f10079"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do0_run1_rew3.npz",
       "bytes": 9261882,
       "sha256": "57287ba933f7388df02cfa814c1b62bb9b96faf0d95fa6c3161272f31b2499ca"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do0_run1_rew6.npz",
       "bytes": 9242782,
       "sha256": "a3ef97ee8069f6929a112e97764c0dbda2b23ab2d6e298f215440be5ad105104"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do1_run1_rew10.npz",
       "bytes": 9194747,
       "sha256": "1c7763f941f503c7d028ded36ecbdd6e2e411551386cf01f9f4bc598ead032ce"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do1_run1_rew12.npz",
       "bytes": 9213622,
       "sha256": "6a107b047e8104f1e958fe29f9a389dd70bab4d34cba11981e40e2f8e38e58e0"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do1_run1_rew15.npz",
       "bytes": 9224375,
       "sha256": "03f94e7622f4ee45d20d955ba6a794724df41568b5bacd1b79550ea222feabd3"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 9191079,
       "sha256": "febb016d1ab20dd0f9694e6a7cb72b4e5a2581f9b41488669adf8ddc21b53b10"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do1_run1_rew20.npz",
       "bytes": 9201908,
       "sha256": "342f78c7a39981f8e6bd7386fcf07c13978bd98498a752370797fafbf2dceaa2"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do1_run1_rew3.npz",
       "bytes": 9199785,
       "sha256": "0457626e8302c081d1fe859c9b2e8eca0ec8017b3ebdc83208b01c40d55edcb9"
      },
      {
       "name": "supercifar_paper_sweep__dg_bbresnet18_do1_run1_rew6.npz",
       "bytes": 9197773,
       "sha256": "bfb1d3d72e1cf722c456a49f56e2163f009c99a5f9158a04ee03b3d358b91793"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do0_run1_rew0.npz",
       "bytes": 11329943,
       "sha256": "fcc8c8e0dd0bec58b939244f2d5d5e05667892714fcd116bc20a3f4b8952370f"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do0_run2_rew0.npz",
       "bytes": 11292350,
       "sha256": "13d90991038172e426915b1f7285aeb20c343e5d0131284f34a5dbeec334b983"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do0_run3_rew0.npz",
       "bytes": 11293700,
       "sha256": "92cd1b60e1bf1672ddfa7bc399714c8cee81101d2c685b620186528d520af9ce"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do0_run4_rew0.npz",
       "bytes": 11306541,
       "sha256": "c67952aeac48258728290d8ca492291f87d68a38c20817eea8296e6d0fffd104"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do0_run5_rew0.npz",
       "bytes": 11343425,
       "sha256": "37bbb76fc279cf915a241bbcef5c48a5e6b02461c7d5eee2cfa25418b3693b1b"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do1_run1_rew0.npz",
       "bytes": 11316481,
       "sha256": "9599a6f07e9e9784dcf7d460441c6b9c4dd8eb602e88cb7049f017ca3de0d5d9"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do1_run2_rew0.npz",
       "bytes": 11268054,
       "sha256": "d4081d2c3658fdf6ef8fe8ee0995e266d06624a873fe17097e0e4c76f4f1048f"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do1_run3_rew0.npz",
       "bytes": 11230479,
       "sha256": "8c49dd4d874ceaafe1889bfaf1712c40d58fdb0a07f632219fa0449ce7effffd"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do1_run4_rew0.npz",
       "bytes": 11272329,
       "sha256": "0e1c2311a490720183e0e0d7844828f0ef8f0b68471cdcef464656fd0aee3456"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__ce_bbresnet18_do1_run5_rew0.npz",
       "bytes": 11251389,
       "sha256": "bab0fa930bac455f1662ada8bc7e199e553146d4ae34ad28bad63e5ede0143fa"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__confidnet_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 151782988,
       "sha256": "f5230f47129b31687d8665daec27dfccf71ae65cd3556df24c4529f4c2797752"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__confidnet_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 137209711,
       "sha256": "905d7a71dfbfbd84f19c2631b28d8312ec0c63f2998ec7627b7ff12c62012beb"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__devries_bbresnet18_do0_run1_rew2.2.npz",
       "bytes": 145353131,
       "sha256": "6d429621f68f016b5da03906b81a645c2310bf9724aaf3ef6b96e60e6412d07a"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__devries_bbresnet18_do1_run1_rew2.2.npz",
       "bytes": 151414444,
       "sha256": "b68a6436c1f55c1e074c537e07315fc341d60ddc1116118b2da44c48696679c6"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do0_run1_rew10.npz",
       "bytes": 152250015,
       "sha256": "8887701895a7cd91b3603741b0440c72de791f412c5f18a94a1a9ece6c21dfe1"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do0_run1_rew12.npz",
       "bytes": 152352260,
       "sha256": "bb3db261dbbc1c1dc1555a8ff39d22771289a461cc94eaca25093b34071a7edc"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do0_run1_rew15.npz",
       "bytes": 152275686,
       "sha256": "6172191fd568ccfa45574cf59fa7e47748b69b2a802b5028538a63ebade3d8ab"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do0_run1_rew20.npz",
       "bytes": 152434201,
       "sha256": "58957ab3d08d53d2d2ea1d6f45f705da3a992b160a50de545ab3f5b0fe24f386"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew10.npz",
       "bytes": 150766518,
       "sha256": "f295073fbb864424cf7ea81f6cfc783641a60d27e41be6fecf0a43efe520c8d9"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew12.npz",
       "bytes": 151171065,
       "sha256": "92638ecc00f58ea4010232decb1ab63e24beb12bcf8eed86785929342abf0423"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew15.npz",
       "bytes": 149045517,
       "sha256": "5344c0504fee63c41893e19cc88cc9712efde06f7f21e4d19b4300003a9ea342"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__dg_bbresnet18_do1_run1_rew20.npz",
       "bytes": 149501461,
       "sha256": "4f0002529ddb158593c4edf1a3c2778254461c99e073ef9874e42232b931927e"
      }
     ]
    }
   },
   "inventory:fourshift_vgg_bridge": {
    "n": 20,
    "extra": [],
    "missing": [],
    "changed": [],
    "other_files": [],
    "sidecars_not_in_freeze_inventory": {
     "n": 20,
     "rule": "recorded, not inventoried by the freeze",
     "files": [
      {
       "name": "cifar100_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2.npz",
       "bytes": 10162814,
       "sha256": "4b4acbe7a8bb2e887221676f8e6f75cae051253deb64462f3bd1d9e71e817f0d"
      },
      {
       "name": "cifar100_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2.npz",
       "bytes": 10158421,
       "sha256": "a084bf279de9a798bfe31bfbd7d50fa836d661cfe7810adba9a5eb5b15c9ed5f"
      },
      {
       "name": "cifar100_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2.npz",
       "bytes": 9292493,
       "sha256": "d75e450d971720139e830e988fe4f6a577139e24a5e56224c07d54284229c5b5"
      },
      {
       "name": "cifar100_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2.npz",
       "bytes": 9219530,
       "sha256": "eef8cd6c425eaca29664ddef3b9a73eae9d9f7c11086ecf1f7bb0474f56da4a2"
      },
      {
       "name": "cifar100_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2.npz",
       "bytes": 8902733,
       "sha256": "fd8da3e974f64467bfa3f3aeb920f2e14f087d912cdc2db20afbd48721b8c1a4"
      },
      {
       "name": "cifar10_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2.npz",
       "bytes": 8020043,
       "sha256": "bdd797987248dc1d49392854beb574d965c2f788565ac17a97e6a67e551d0ebe"
      },
      {
       "name": "cifar10_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2.npz",
       "bytes": 7025046,
       "sha256": "cdea67836f49ad2a1421da385c3feae4e4b157a6ab637ec120b0a7a190968766"
      },
      {
       "name": "cifar10_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2.npz",
       "bytes": 7873681,
       "sha256": "fc7000f731171bc26d68caabefc11752f7784bb39757e4fecf78558743f40af3"
      },
      {
       "name": "cifar10_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2.npz",
       "bytes": 3826710,
       "sha256": "edc990e4479fd494df0515c20ba8527413dba7c48d08f72adf3a52868f95b4f7"
      },
      {
       "name": "cifar10_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2.npz",
       "bytes": 7320762,
       "sha256": "256a561e201bf2b6484ee06ed9ebd13437d933e9a8b09bacd02aed649939142b"
      },
      {
       "name": "supercifar_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2.npz",
       "bytes": 8250593,
       "sha256": "4acbd625c9ae0cd21013066408e3e8072b148bfc3efbdeb54c0d2c00c512f5e0"
      },
      {
       "name": "supercifar_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2.npz",
       "bytes": 7648956,
       "sha256": "14c00492d9f9092528c26b1fcf6afdc4c397f31d3aeb99bf24be137e970d42dc"
      },
      {
       "name": "supercifar_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2.npz",
       "bytes": 6888060,
       "sha256": "c6477606a4ea7c065c92f006d9dd807f0a34e272e0f9aa4646e85cfa69fd0070"
      },
      {
       "name": "supercifar_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2.npz",
       "bytes": 7983543,
       "sha256": "21ab6df9bd9989ce6ff66b878a1bc747ae70fc074254b982727e34dc3d24c67c"
      },
      {
       "name": "supercifar_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2.npz",
       "bytes": 7718783,
       "sha256": "341472a066f5b93d73e8c71efa16082e435489818d4e6962fec957609d515ca9"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2.npz",
       "bytes": 151742601,
       "sha256": "e91b70a511cd83a05c82e694ff07de2caf0a8a3ae2e31912f0eb126c6453a47f"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2.npz",
       "bytes": 151808618,
       "sha256": "84e017afd0eb95f75b672a9d565ee5b46a75209dceaf52d5109262b834d6148d"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2.npz",
       "bytes": 151849804,
       "sha256": "63c4757ab1817d6f758e5ca882145344ccf3211b60ef489bd60ee4506a6c6f8b"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2.npz",
       "bytes": 151619504,
       "sha256": "fbe6695194a36d87b20f51729284455c2b6eb3b75003fa64b64cb34482b12267"
      },
      {
       "name": "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2.npz",
       "bytes": 151739019,
       "sha256": "639ad88c4b1a5a246b5de5faf840b0d60836fc035d5cd03fba5486d84a808dd2"
      }
     ]
    }
   },
   "licenses_v1": {
    "SEL_Nf10": 1.1,
    "SEL_Nf5": 1.1,
    "LEVEL_Nf10": 1.1,
    "LEVEL_Nf5": 1.1
   },
   "license_v2": {
    "path": "rn18_handoff_replication/simulations/qualification_report_v2.json",
    "sha256": "5a7264c58aa50e134ca7c456ffed0351bc906ec45a9c6f49a0ca2ff48872fddb",
    "seed": 2404,
    "licensed": {
     "SEL_Nf5": 2.0
    },
    "unlicensed": {
     "SEL_Nf10": [
      "metric_almost_all_ties"
     ],
     "LEVEL_Nf10": [
      "null_level_bounded_beta_family"
     ],
     "LEVEL_Nf5": [
      "null_level_bounded_beta_family"
     ]
    }
   },
   "key_set": {
    "n_expected": 384,
    "n_present": 384,
    "missing": 0,
    "extra": 0,
    "bad_fields": 0
   },
   "families": {
    "n": 10,
    "checkpoints_per_family": 4,
    "seed_formula": "270000 + 5*dropout + (run-1)"
   },
   "phase1_consistency": {
    "n": 96,
    "tolerance_rel": 1e-06
   },
   "severity_axes": {
    "n_sources": 4,
    "n_keys": 32
   },
   "denominators": {
    "rn18_records": 96,
    "rn18_cells": 384,
    "ce_families": 10,
    "ce_do0_families": 5,
    "vgg_bridge_records": 20,
    "sets_per_source": 4
   }
  },
  "verdict": "VALIDATION PASSED"
 },
 "denominators": {
  "n_records": 96,
  "n_cells": 384,
  "per_source_checkpoints": {
   "cifar10": 22,
   "cifar100": 24,
   "supercifar100": 28,
   "tinyimagenet": 22
  },
  "ce_families": [
   "do0_run1",
   "do0_run2",
   "do0_run3",
   "do0_run4",
   "do0_run5",
   "do1_run1",
   "do1_run2",
   "do1_run3",
   "do1_run4",
   "do1_run5"
  ],
  "vgg_checkpoints_primary_view": 20,
  "vgg_checkpoints_aug_view": 20,
  "multipliers_v2": {
   "SEL_Nf10": null,
   "SEL_Nf5": 2.0,
   "LEVEL_Nf10": null,
   "LEVEL_Nf5": null
  },
  "licenses_v2": {
   "SEL_Nf10": {
    "licensed": false,
    "multiplier": null,
    "reason": "no multiplier passed development",
    "failed_scenarios": [
     "metric_almost_all_ties"
    ]
   },
   "SEL_Nf5": {
    "licensed": true,
    "multiplier": 2.0,
    "failed_scenarios": []
   },
   "LEVEL_Nf10": {
    "licensed": false,
    "multiplier": null,
    "reason": "no multiplier passed development",
    "failed_scenarios": [
     "null_level_bounded_beta_family"
    ]
   },
   "LEVEL_Nf5": {
    "licensed": false,
    "multiplier": null,
    "reason": "no multiplier passed development",
    "failed_scenarios": [
     "null_level_bounded_beta_family"
    ]
   }
  }
 },
 "HO": {
  "dK": {
   "cifar10": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033999999999999994,
      "g_at_max_d": 0.0050214285714285725,
      "band_q95": 0.002091642857142856,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0009892857142857145,
      "g_at_max_d": 0.0067800000000000004,
      "band_q95": 0.0032288520893403267,
      "tie_region": [
       -1.57,
       0.813
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0010362499999999998,
      "g_at_max_d": 0.011903750000000001,
      "band_q95": 0.0023453124999999997,
      "tie_region": [
       -1.57,
       -0.776
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 4,
      "dg/do1": 1
     },
     "middle": {
      "ce/do0": 2,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 3
     },
     "weak": {
      "ce/do0": 3,
      "ce/do1": 5
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "cifar100": {
    "n_checkpoints": 24,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 8,
     "middle": 8,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033768749999999997,
      "g_at_max_d": 0.005915,
      "band_q95": 0.003268125,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00162375,
      "g_at_max_d": 0.007115,
      "band_q95": 0.003537343806981895,
      "tie_region": [
       -1.271,
       0.064
      ]
     },
     "weak": {
      "n_sign_changes": 1,
      "all_crossings": [
       -0.331
      ],
      "first_up_crossing": -0.331,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.00444625,
      "g_at_max_d": 0.0076237499999999995,
      "band_q95": 0.007247625,
      "tie_region": [
       -1.271,
       1.002
      ]
     }
    },
    "informative": true,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     24,
     24
    ],
    "composition": {
     "strong": {
      "ce/do0": 1,
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 5
     },
     "middle": {
      "ce/do0": 4,
      "ce/do1": 4
     },
     "weak": {
      "ce/do1": 1,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 5
     }
    },
    "verdict": "HO-RETAINED",
    "within_dg": {
     "retained": true,
     "informative": true,
     "sizes": {
      "strong": 3,
      "middle": 3,
      "weak": 4
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "supercifar100": {
    "n_checkpoints": 28,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 9,
     "middle": 9,
     "weak": 10
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": 0.002418611111111112,
      "g_at_max_d": 0.002418611111111112,
      "band_q95": 0.0021747777777777785,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00258,
      "g_at_max_d": 0.006938888888888889,
      "band_q95": 0.003126916666666667,
      "tie_region": [
       -1.273,
       -0.487
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0024514999999999997,
      "g_at_max_d": 0.012273999999999998,
      "band_q95": 0.005215449999999999,
      "tie_region": [
       -1.273,
       -0.163
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     28,
     28
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 7
     },
     "middle": {
      "ce/do0": 4,
      "dg/do1": 5
     },
     "weak": {
      "ce/do0": 1,
      "ce/do1": 5,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 2
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "retained": true,
     "informative": false,
     "sizes": {
      "strong": 4,
      "middle": 5,
      "weak": 5
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   },
   "tinyimagenet": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.003829642857142857,
      "g_at_max_d": 0.003829642857142857,
      "band_q95": 0.0026179761904761914,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0016114285714285714,
      "g_at_max_d": 0.007874285714285714,
      "band_q95": 0.0049351666666666676,
      "tie_region": [
       -1.611,
       -0.348
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.006830937499999999,
      "g_at_max_d": -0.006830937499999999,
      "band_q95": 0.0095135625,
      "tie_region": [
       -1.611,
       0.882
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "ce/do0": 5,
      "confidnet/do0": 1,
      "dg/do0": 1
     },
     "middle": {
      "ce/do1": 5,
      "devries/do0": 1,
      "dg/do0": 1
     },
     "weak": {
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do0": 2,
      "dg/do1": 4
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   }
  },
  "dF": {
   "cifar10": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033999999999999994,
      "g_at_max_d": 0.0050214285714285725,
      "band_q95": 0.002091642857142856,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0009892857142857145,
      "g_at_max_d": 0.0067800000000000004,
      "band_q95": 0.0032280094692150817,
      "tie_region": [
       -1.684,
       0.597
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0010362499999999998,
      "g_at_max_d": 0.011903750000000001,
      "band_q95": 0.0023453124999999997,
      "tie_region": [
       -1.684,
       -0.586
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 4,
      "dg/do1": 1
     },
     "middle": {
      "ce/do0": 2,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 3
     },
     "weak": {
      "ce/do0": 3,
      "ce/do1": 5
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "cifar100": {
    "n_checkpoints": 24,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 8,
     "middle": 8,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0033768749999999997,
      "g_at_max_d": 0.005915,
      "band_q95": 0.003268125,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00162375,
      "g_at_max_d": 0.007115,
      "band_q95": 0.003546769802021209,
      "tie_region": [
       -1.541,
       0.157
      ]
     },
     "weak": {
      "n_sign_changes": 1,
      "all_crossings": [
       0.018
      ],
      "first_up_crossing": 0.018,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.00444625,
      "g_at_max_d": 0.0076237499999999995,
      "band_q95": 0.007247625,
      "tie_region": [
       -1.541,
       1.056
      ]
     }
    },
    "informative": true,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     24,
     24
    ],
    "composition": {
     "strong": {
      "ce/do0": 1,
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 5
     },
     "middle": {
      "ce/do0": 4,
      "ce/do1": 4
     },
     "weak": {
      "ce/do1": 1,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 5
     }
    },
    "verdict": "HO-RETAINED",
    "within_dg": {
     "retained": true,
     "informative": true,
     "sizes": {
      "strong": 3,
      "middle": 3,
      "weak": 4
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": true
    }
   },
   "supercifar100": {
    "n_checkpoints": 28,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 9,
     "middle": 9,
     "weak": 10
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": 0.002418611111111112,
      "g_at_max_d": 0.002418611111111112,
      "band_q95": 0.0021747777777777785,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.00258,
      "g_at_max_d": 0.006938888888888889,
      "band_q95": 0.003126916666666667,
      "tie_region": [
       -1.531,
       -0.041
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0024514999999999997,
      "g_at_max_d": 0.012273999999999998,
      "band_q95": 0.005215449999999999,
      "tie_region": [
       -1.531,
       0.069
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     28,
     28
    ],
    "composition": {
     "strong": {
      "confidnet/do0": 1,
      "devries/do0": 1,
      "dg/do0": 7
     },
     "middle": {
      "ce/do0": 4,
      "dg/do1": 5
     },
     "weak": {
      "ce/do0": 1,
      "ce/do1": 5,
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do1": 2
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "retained": true,
     "informative": false,
     "sizes": {
      "strong": 4,
      "middle": 5,
      "weak": 5
     }
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   },
   "tinyimagenet": {
    "n_checkpoints": 22,
    "n_sets": 4,
    "tertile_sizes": {
     "strong": 7,
     "middle": 7,
     "weak": 8
    },
    "full_suite": {
     "strong": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.003829642857142857,
      "g_at_max_d": 0.003829642857142857,
      "band_q95": 0.002718130952380952,
      "tie_region": null
     },
     "middle": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": false,
      "g_at_min_d": 0.0016114285714285714,
      "g_at_max_d": 0.006416190476190476,
      "band_q95": 0.0038056869782507725,
      "tie_region": [
       -1.64,
       -0.863
      ]
     },
     "weak": {
      "n_sign_changes": 0,
      "all_crossings": [],
      "first_up_crossing": null,
      "bracketed_by_observed": true,
      "g_at_min_d": -0.006830937499999999,
      "g_at_max_d": -0.006830937499999999,
      "band_q95": 0.008032916666666664,
      "tie_region": [
       -1.64,
       0.98
      ]
     }
    },
    "informative": false,
    "retained_full": true,
    "single_shift_deletions": {
     "fashionmnist_new": true,
     "kmnist_new": true,
     "mnist_new": true,
     "stl10_new": true
    },
    "n_checkpoint_deletions_retained": [
     22,
     22
    ],
    "composition": {
     "strong": {
      "ce/do0": 5,
      "confidnet/do0": 1,
      "dg/do0": 1
     },
     "middle": {
      "ce/do1": 5,
      "devries/do0": 1,
      "dg/do0": 1
     },
     "weak": {
      "confidnet/do1": 1,
      "devries/do1": 1,
      "dg/do0": 2,
      "dg/do1": 4
     }
    },
    "verdict": "HO-UNINFORMATIVE",
    "within_dg": {
     "verdict": "UNINFORMATIVE (< 9 DG checkpoints)"
    },
    "within_ce_dropout": {
     "lower_nc1_dropout": 0,
     "retained": true,
     "informative": false
    }
   }
  },
  "global": {
   "retained_sources_dK": [
    "cifar100"
   ],
   "retained_sources_dF": [
    "cifar100"
   ],
   "global_wording_available": false,
   "attribution": "compatible with geometry organizing the handoff and equally with the training objective doing so; the design does not separate them"
  },
  "evidence_class": "REGISTERED (gate-based, no alpha)"
 },
 "SEL_ce": {
  "evidence_class": "DESCRIPTIVE (family unlicensed by the version-2 qualification; interval at multiplier 1 is conditional on the approximate procedure)",
  "declared_class": "REGISTERED (alpha 0.025, Bonferroni 9)",
  "license": {
   "licensed": false,
   "failed_scenarios": [
    "metric_almost_all_ties"
   ]
  },
  "N_f": 10,
  "n_cells": 160,
  "multiplier": 1.0,
  "mean_regret_P00": 0.00898875,
  "P00_choice_counts": {
   "CTM": 160,
   "Energy": 0,
   "tie": 0
  },
  "P00_identical_to_always_ctm": true,
  "per_source_mean_regret_P00": {
   "cifar10": 0.0008375000000000021,
   "cifar100": 0.0070600000000000055,
   "supercifar100": 0.0122975,
   "tinyimagenet": 0.015759999999999993
  },
  "material_subset": {
   "n_material": 17,
   "n_all": 160,
   "sign_accuracy_material_dG": 1.0,
   "sign_accuracy_all_nonzero_dG": 0.86875
  },
  "comparators": {
   "always_energy": {
    "mean_regret": 0.015504999999999996,
    "D_b": 0.006516249999999996,
    "se": 0.003428340093778783,
    "ci": [
     -0.0074550334734147985,
     0.02048753347341479
    ],
    "ci_display": [
     -0.00746,
     0.02049
    ],
    "not_estimable": null
   },
   "always_ctm": {
    "mean_regret": 0.00898875,
    "D_b": 0.0,
    "se": 0.0,
    "ci": null,
    "ci_display": null,
    "not_estimable": "zero or non-finite jackknife standard error"
   },
   "vgg_kid_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "se": 0.0012800998224964928,
    "ci": [
     -0.0023767045873951415,
     0.008056704587395139
    ],
    "ci_display": [
     -0.00238,
     0.00806
    ],
    "not_estimable": null
   },
   "vgg_fd_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "se": 0.0012800998224964928,
    "ci": [
     -0.0023767045873951415,
     0.008056704587395139
    ],
    "ci_display": [
     -0.00238,
     0.00806
    ],
    "not_estimable": null
   },
   "vgg_source_shift_mean": {
    "mean_regret": 0.011768749999999998,
    "D_b": 0.0027799999999999978,
    "se": 0.0013693250314175476,
    "ci": [
     -0.0028003180716795143,
     0.00836031807167951
    ],
    "ci_display": [
     -0.0028,
     0.00836
    ],
    "not_estimable": null
   },
   "vgg_geometry_severity_ridge": {
    "mean_regret": 0.008616875000000001,
    "D_b": -0.00037187499999999894,
    "se": 0.00043643247735665366,
    "ci": [
     -0.0021504388797094023,
     0.0014066888797094044
    ],
    "ci_display": [
     -0.00215,
     0.00141
    ],
    "not_estimable": null
   },
   "vgg_matched_scalar_ridge": {
    "mean_regret": 0.009133124999999999,
    "D_b": 0.00014437499999999867,
    "se": 0.0003814860949300779,
    "ci": [
     -0.0014102695882384688,
     0.0016990195882384661
    ],
    "ci_display": [
     -0.00141,
     0.0017
    ],
    "not_estimable": null
   },
   "vgg_no_target_batch_ridge": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "se": 0.0013965959514963924,
    "ci": [
     4.91715845155425e-05,
     0.011432078415484456
    ],
    "ci_display": [
     5e-05,
     0.01143
    ],
    "not_estimable": null
   },
   "vgg_source_majority": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "se": 0.0013965959514963924,
    "ci": [
     4.91715845155425e-05,
     0.011432078415484456
    ],
    "ci_display": [
     5e-05,
     0.01143
    ],
    "not_estimable": null
   }
  },
  "verdict": "NO LICENSE: descriptive",
  "verdict_if_licensed_at_multiplier_1": "PRACTICALLY EQUIVALENT TO THE REFERENCE"
 },
 "SEL_ce_do0_sensitivity_Nf5": {
  "evidence_class": "PRE-SPECIFIED SENSITIVITY (descriptive)",
  "license": {
   "licensed": true,
   "multiplier": 2.0
  },
  "N_f": 5,
  "n_cells": 80,
  "multiplier": 2.0,
  "mean_regret_P00": 0.010761250000000002,
  "P00_choice_counts": {
   "CTM": 80,
   "Energy": 0,
   "tie": 0
  },
  "P00_identical_to_always_ctm": true,
  "per_source_mean_regret_P00": {
   "cifar10": 0.0,
   "cifar100": 0.007665000000000005,
   "supercifar100": 0.01583,
   "tinyimagenet": 0.01955
  },
  "material_subset": {
   "n_material": 4,
   "n_all": 80,
   "sign_accuracy_material_dG": 1.0,
   "sign_accuracy_all_nonzero_dG": 0.8875
  },
  "comparators": {
   "always_energy": {
    "mean_regret": 0.010666249999999999,
    "D_b": -9.500000000000307e-05,
    "se": 0.0039256734095884736,
    "ci": [
     -0.05167599694771695,
     0.05148599694771694
    ],
    "ci_display": [
     -0.05168,
     0.05149
    ],
    "not_estimable": null
   },
   "always_ctm": {
    "mean_regret": 0.010761250000000002,
    "D_b": 0.0,
    "se": 0.0,
    "ci": null,
    "ci_display": null,
    "not_estimable": "zero or non-finite jackknife standard error"
   },
   "vgg_kid_isotonic": {
    "mean_regret": 0.012115,
    "D_b": 0.001353749999999999,
    "se": 0.0011460727780119367,
    "ci": [
     -0.0137049597541291,
     0.016412459754129097
    ],
    "ci_display": [
     -0.0137,
     0.01641
    ],
    "not_estimable": null
   },
   "vgg_fd_isotonic": {
    "mean_regret": 0.012115,
    "D_b": 0.001353749999999999,
    "se": 0.0011460727780119367,
    "ci": [
     -0.0137049597541291,
     0.016412459754129097
    ],
    "ci_display": [
     -0.0137,
     0.01641
    ],
    "not_estimable": null
   },
   "vgg_source_shift_mean": {
    "mean_regret": 0.011656250000000002,
    "D_b": 0.000895,
    "se": 0.0011694473187151258,
    "ci": [
     -0.014470837216571747,
     0.016260837216571747
    ],
    "ci_display": [
     -0.01447,
     0.01626
    ],
    "not_estimable": null
   },
   "vgg_geometry_severity_ridge": {
    "mean_regret": 0.011380000000000005,
    "D_b": 0.000618750000000003,
    "se": 0.0005084750485520402,
    "ci": [
     -0.006062307538635755,
     0.007299807538635761
    ],
    "ci_display": [
     -0.00606,
     0.0073
    ],
    "not_estimable": null
   },
   "vgg_matched_scalar_ridge": {
    "mean_regret": 0.011145,
    "D_b": 0.0003837499999999987,
    "se": 0.0007856317123818284,
    "ci": [
     -0.009938980072098588,
     0.010706480072098585
    ],
    "ci_display": [
     -0.00994,
     0.01071
    ],
    "not_estimable": null
   },
   "vgg_no_target_batch_ridge": {
    "mean_regret": 0.013905000000000004,
    "D_b": 0.0031437500000000024,
    "se": 0.001201481832467725,
    "ci": [
     -0.012643001537171744,
     0.018930501537171747
    ],
    "ci_display": [
     -0.01264,
     0.01893
    ],
    "not_estimable": null
   },
   "vgg_source_majority": {
    "mean_regret": 0.013905000000000004,
    "D_b": 0.0031437500000000024,
    "se": 0.001201481832467725,
    "ci": [
     -0.012643001537171744,
     0.018930501537171747
    ],
    "ci_display": [
     -0.01264,
     0.01893
    ],
    "not_estimable": null
   }
  },
  "verdict": "UNRESOLVED"
 },
 "SEL_ce_augview_comparators_sensitivity": {
  "evidence_class": "DESCRIPTIVE (family unlicensed by the version-2 qualification; interval at multiplier 1 is conditional on the approximate procedure)",
  "declared_class": "PRE-SPECIFIED SENSITIVITY (descriptive)",
  "license": {
   "licensed": false,
   "failed_scenarios": [
    "metric_almost_all_ties"
   ]
  },
  "N_f": 10,
  "n_cells": 160,
  "multiplier": 1.0,
  "mean_regret_P00": 0.00898875,
  "P00_choice_counts": {
   "CTM": 160,
   "Energy": 0,
   "tie": 0
  },
  "P00_identical_to_always_ctm": true,
  "per_source_mean_regret_P00": {
   "cifar10": 0.0008375000000000021,
   "cifar100": 0.0070600000000000055,
   "supercifar100": 0.0122975,
   "tinyimagenet": 0.015759999999999993
  },
  "material_subset": {
   "n_material": 17,
   "n_all": 160,
   "sign_accuracy_material_dG": 1.0,
   "sign_accuracy_all_nonzero_dG": 0.86875
  },
  "comparators": {
   "always_energy": {
    "mean_regret": 0.015504999999999996,
    "D_b": 0.006516249999999996,
    "se": 0.003428340093778783,
    "ci": [
     -0.0074550334734147985,
     0.02048753347341479
    ],
    "ci_display": [
     -0.00746,
     0.02049
    ],
    "not_estimable": null
   },
   "always_ctm": {
    "mean_regret": 0.00898875,
    "D_b": 0.0,
    "se": 0.0,
    "ci": null,
    "ci_display": null,
    "not_estimable": "zero or non-finite jackknife standard error"
   },
   "vgg_kid_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "se": 0.0012800998224964928,
    "ci": [
     -0.0023767045873951415,
     0.008056704587395139
    ],
    "ci_display": [
     -0.00238,
     0.00806
    ],
    "not_estimable": null
   },
   "vgg_fd_isotonic": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "se": 0.0012800998224964928,
    "ci": [
     -0.0023767045873951415,
     0.008056704587395139
    ],
    "ci_display": [
     -0.00238,
     0.00806
    ],
    "not_estimable": null
   },
   "vgg_source_shift_mean": {
    "mean_regret": 0.011828749999999999,
    "D_b": 0.0028399999999999988,
    "se": 0.0012800998224964928,
    "ci": [
     -0.0023767045873951415,
     0.008056704587395139
    ],
    "ci_display": [
     -0.00238,
     0.00806
    ],
    "not_estimable": null
   },
   "vgg_geometry_severity_ridge": {
    "mean_regret": 0.0087425,
    "D_b": -0.00024624999999999994,
    "se": 0.0005204800468477789,
    "ci": [
     -0.0023673272787572077,
     0.0018748272787572078
    ],
    "ci_display": [
     -0.00237,
     0.00187
    ],
    "not_estimable": null
   },
   "vgg_matched_scalar_ridge": {
    "mean_regret": 0.0073693749999999975,
    "D_b": -0.0016193750000000028,
    "se": 0.0014876632292560553,
    "ci": [
     -0.007681948758836997,
     0.004443198758836991
    ],
    "ci_display": [
     -0.00768,
     0.00444
    ],
    "not_estimable": null
   },
   "vgg_no_target_batch_ridge": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "se": 0.0013965959514963924,
    "ci": [
     4.91715845155425e-05,
     0.011432078415484456
    ],
    "ci_display": [
     5e-05,
     0.01143
    ],
    "not_estimable": null
   },
   "vgg_source_majority": {
    "mean_regret": 0.014729375,
    "D_b": 0.005740624999999999,
    "se": 0.0013965959514963924,
    "ci": [
     4.91715845155425e-05,
     0.011432078415484456
    ],
    "ci_display": [
     5e-05,
     0.01143
    ],
    "not_estimable": null
   }
  },
  "verdict": "NO LICENSE: descriptive",
  "verdict_if_licensed_at_multiplier_1": "UNRESOLVED"
 },
 "SEL_paradigm_pool_descriptive": {
  "evidence_class": "DESCRIPTIVE",
  "mean_regret_P00": 0.012820982142857141,
  "n_nonfinite": 0
 },
 "LEVEL_ce": {
  "evidence_class": "DESCRIPTIVE (family unlicensed by the version-2 qualification; interval at multiplier 1 is conditional on the approximate procedure)",
  "declared_class": "REGISTERED (alpha 0.025)",
  "license": {
   "licensed": false,
   "failed_scenarios": [
    "null_level_bounded_beta_family"
   ]
  },
  "multiplier": 1.0,
  "N_f": 10,
  "delta": 0.011144496547958066,
  "se": 0.0013164806247980185,
  "family_deltas": {
   "do0_run1": 0.007874190709179552,
   "do0_run2": 0.007130912955558566,
   "do0_run3": 0.007355957577627359,
   "do0_run4": 0.007770360636993878,
   "do0_run5": 0.0070130142315102395,
   "do1_run1": 0.017361103736612815,
   "do1_run2": 0.01348715371086262,
   "do1_run3": 0.012748916963831236,
   "do1_run4": 0.016811306831227113,
   "do1_run5": 0.013892048126177281
  },
  "ci": [
   0.007609731790751679,
   0.014679261305164453
  ],
  "ci_display": [
   0.00761,
   0.01468
  ],
  "verdict": "NO LICENSE: descriptive",
  "equivalent_within_0.01": false,
  "at_least_one_point": false,
  "verdict_if_licensed_at_multiplier_1": "resolved improvement"
 },
 "LEVEL_ce_do0_sensitivity_Nf5": {
  "evidence_class": "DESCRIPTIVE (family unlicensed by the version-2 qualification; interval at multiplier 1 is conditional on the approximate procedure)",
  "declared_class": "PRE-SPECIFIED SENSITIVITY (descriptive)",
  "license": {
   "licensed": false,
   "failed_scenarios": [
    "null_level_bounded_beta_family"
   ]
  },
  "multiplier": 1.0,
  "N_f": 5,
  "delta": 0.0074288872221739185,
  "se": 0.00017058092354515751,
  "family_deltas": {
   "do0_run1": 0.007874190709179552,
   "do0_run2": 0.007130912955558566,
   "do0_run3": 0.007355957577627359,
   "do0_run4": 0.007770360636993878,
   "do0_run5": 0.0070130142315102395
  },
  "ci": [
   0.006832637650040042,
   0.008025136794307795
  ],
  "ci_display": [
   0.00683,
   0.00803
  ],
  "verdict": "NO LICENSE: descriptive",
  "equivalent_within_0.01": true,
  "at_least_one_point": false,
  "verdict_if_licensed_at_multiplier_1": "resolved improvement"
 },
 "bridge_A_G": {
  "evidence_class": "DESCRIPTIVE (consumed VGG bridge)",
  "primary_view": -0.00125575,
  "aug_view": -0.001478
 },
 "ORG_descriptive": {
  "evidence_class": "DESCRIPTIVE",
  "ce_do0": {
   "A_G": 0.0018057499999999996,
   "ci95_descriptive": [
    -0.00019750112911534352,
    0.0038090011291153428
   ],
   "ci95_display": [
    -0.0002,
    0.00381
   ],
   "N_f": 5,
   "not_estimable": null,
   "equivalence_0.003": false,
   "resolvable": true
  },
  "full_panel_descriptive": {
   "A_G": 0.0032976609209680874,
   "caveat": "geometry and training objective move together on this panel"
  }
 },
 "E4": {
  "evidence_class": "DESCRIPTIVE",
  "full": {
   "spearman_absM_absdG_all": -0.1427845038756117,
   "spearman_absM_absdG_material": -0.252716107797133
  },
  "ce": {
   "spearman_absM_absdG_all": -0.11171034446243945,
   "spearman_absM_absdG_material": -0.047823430202802525
  }
 },
 "comparator_fits": {
  "primary_view": {
   "vgg_geometry_severity_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.001229238799443499,
     "0.001": 0.0012292347510209564,
     "0.01": 0.0012291943078959686,
     "0.1": 0.0012287939425876765,
     "1.0": 0.001225157205964494,
     "10.0": 0.0012066635433142072
    }
   },
   "vgg_matched_scalar_ridge": {
    "lambda": 0.1,
    "cv_losses": {
     "0.0001": 0.0005528502577014881,
     "0.001": 0.0005499529320381879,
     "0.01": 0.0005472454632546573,
     "0.1": 0.0005459801704182715,
     "1.0": 0.0005547223148980342,
     "10.0": 0.0008073791713627316
    }
   },
   "vgg_no_target_batch_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.0013592237549551226,
     "0.001": 0.0013524770767664946,
     "0.01": 0.0013377710427991954,
     "0.1": 0.0013214045607284008,
     "1.0": 0.001307312481862802,
     "10.0": 0.0013029024805854376
    }
   },
   "source_majority": {
    "cifar10": -1.0,
    "cifar100": 1.0,
    "supercifar100": -1.0,
    "tinyimagenet": 1.0
   },
   "folds": "leave-one-VGG-checkpoint-out (seed reuse across sources not audited)"
  },
  "aug_view": {
   "vgg_geometry_severity_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.001230544602407013,
     "0.001": 0.001230541718425619,
     "0.01": 0.0012305129070332606,
     "0.1": 0.001230227605741492,
     "1.0": 0.001227629249961061,
     "10.0": 0.0012143770180766999
    }
   },
   "vgg_matched_scalar_ridge": {
    "lambda": 0.0001,
    "cv_losses": {
     "0.0001": 0.0006202676805690363,
     "0.001": 0.0006210132569474233,
     "0.01": 0.0006238369262680396,
     "0.1": 0.0006302314737357216,
     "1.0": 0.0006635855569136133,
     "10.0": 0.0009490327833837318
    }
   },
   "vgg_no_target_batch_ridge": {
    "lambda": 10.0,
    "cv_losses": {
     "0.0001": 0.001429376755753756,
     "0.001": 0.0014141538274670407,
     "0.01": 0.0013910020809534632,
     "0.1": 0.0013843656088782558,
     "1.0": 0.0013833599157311235,
     "10.0": 0.0013822624997759164
    }
   },
   "source_majority": {
    "cifar10": -1.0,
    "cifar100": 1.0,
    "supercifar100": -1.0,
    "tinyimagenet": 1.0
   },
   "folds": "leave-one-VGG-checkpoint-out (seed reuse across sources not audited)"
  }
 },
 "comparator_specification": {
  "primary_view": {
   "folds": {
    "rule": "leave-one-VGG-checkpoint-out",
    "fold_ids": [
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2"
    ]
   },
   "indicator_sources": [
    "cifar100",
    "supercifar100",
    "tinyimagenet"
   ],
   "reference_source": "cifar10",
   "ridge": {
    "vgg_geometry_severity_ridge": {
     "features": [
      "dK",
      "g_pct",
      "dK*g_pct"
     ],
     "lambda": 10.0,
     "cv_losses": {
      "0.0001": 0.001229238799443499,
      "0.001": 0.0012292347510209564,
      "0.01": 0.0012291943078959686,
      "0.1": 0.0012287939425876765,
      "1.0": 0.001225157205964494,
      "10.0": 0.0012066635433142072
     },
     "scaler_mean": [
      -3.219646771412954e-16,
      0.5,
      -1.1102230246251565e-16
     ],
     "scaler_sd": [
      0.9999999967294846,
      0.282842712474619,
      0.5744562627750348
     ],
     "beta": {
      "intercept": -0.0023400000000000235,
      "source_indicators": {
       "cifar100": 0.06774000000000004,
       "supercifar100": -0.005984999999999987,
       "tinyimagenet": 0.04136500000000002
      },
      "standardized_features": {
       "dK": 0.00553025557347682,
       "g_pct": 0.002356236929753842,
       "dK*g_pct": 0.006464510186283771
      }
     }
    },
    "vgg_matched_scalar_ridge": {
     "features": [
      "logC",
      "logD",
      "logNC1",
      "s_dict",
      "theta_deg",
      "logit",
      "eta",
      "log_gamma",
      "a",
      "log_rho"
     ],
     "lambda": 0.1,
     "cv_losses": {
      "0.0001": 0.0005528502577014881,
      "0.001": 0.0005499529320381879,
      "0.01": 0.0005472454632546573,
      "0.1": 0.0005459801704182715,
      "1.0": 0.0005547223148980342,
      "10.0": 0.0008073791713627316
     },
     "scaler_mean": [
      3.787627906174154,
      6.584898215319481,
      -1.6055719370069248,
      15.80528979059713,
      15.154947272718232,
      11.766702844400541,
      0.04055815262810172,
      -0.8525175402188058,
      0.5419124977130482,
      0.29936926233968003
     ],
     "scaler_sd": [
      1.2110819283366436,
      0.6002830669264719,
      0.8841274134482827,
      8.211192312368237,
      4.5806576728205055,
      1.542895469132876,
      0.02368345203564554,
      0.5148245458334679,
      0.111356636191316,
      0.2870805831078862
     ],
     "beta": {
      "intercept": -0.07323802718534471,
      "source_indicators": {
       "cifar100": 0.09000446628456049,
       "supercifar100": 0.0529947510822305,
       "tinyimagenet": 0.24371289137458774
      },
      "standardized_features": {
       "logC": -1.2652649492528666e-14,
       "logD": 1.3032496796564949e-14,
       "logNC1": 0.024627811241071444,
       "s_dict": -0.040554590696907936,
       "theta_deg": 0.008753204775495694,
       "logit": -0.00238173004258995,
       "eta": 0.009569406137459458,
       "log_gamma": 0.06035556380600435,
       "a": -0.003772233206756292,
       "log_rho": 0.08017230735034049
      }
     }
    },
    "vgg_no_target_batch_ridge": {
     "features": [
      "logC",
      "logD",
      "logNC1",
      "s_dict",
      "theta_deg",
      "logit",
      "eta"
     ],
     "lambda": 10.0,
     "cv_losses": {
      "0.0001": 0.0013592237549551226,
      "0.001": 0.0013524770767664946,
      "0.01": 0.0013377710427991954,
      "0.1": 0.0013214045607284008,
      "1.0": 0.001307312481862802,
      "10.0": 0.0013029024805854376
     },
     "scaler_mean": [
      3.787627906174154,
      6.584898215319481,
      -1.6055719370069248,
      15.80528979059713,
      15.154947272718232,
      11.766702844400541,
      0.04055815262810172
     ],
     "scaler_sd": [
      1.2110819283366436,
      0.6002830669264719,
      0.8841274134482827,
      8.211192312368237,
      4.5806576728205055,
      1.542895469132876,
      0.02368345203564554
     ],
     "beta": {
      "intercept": -0.002323932681688794,
      "source_indicators": {
       "cifar100": 0.06711934815833856,
       "supercifar100": -0.006326827237652996,
       "tinyimagenet": 0.04226320980606959
      },
      "standardized_features": {
       "logC": -8.223442294772423e-17,
       "logD": 1.050234278960286e-16,
       "logNC1": 0.0004627557507864602,
       "s_dict": -0.0006407178628462224,
       "theta_deg": -0.0001370381487767461,
       "logit": -3.4820580355428084e-05,
       "eta": -0.00010358424621689642
      }
     }
    }
   },
   "isotonic": {
    "vgg_kid_isotonic": {
     "cifar10": {
      "d": [
       -1.570392530375779,
       -0.16014242666029932,
       0.7916262237845381,
       0.9389087332515403
      ],
      "fitted_gap": [
       -0.003773333333333332,
       -0.003773333333333332,
       -0.003773333333333332,
       0.001959999999999984
      ]
     },
     "cifar100": {
      "d": [
       -1.2706142527915594,
       -0.6853171993881652,
       0.9389413749888746,
       1.016990077190847
      ],
      "fitted_gap": [
       0.004499999999999993,
       0.07934,
       0.08887999999999999,
       0.08887999999999999
      ]
     },
     "supercifar100": {
      "d": [
       -1.2732109525965163,
       -0.6802831009108223,
       0.9131890670898313,
       1.0403049864175054
      ],
      "fitted_gap": [
       -0.009080000000000013,
       -0.009080000000000013,
       -0.009080000000000013,
       -0.00606000000000001
      ]
     },
     "tinyimagenet": {
      "d": [
       -1.6112361310487624,
       -0.059736111806425136,
       0.7892498256123917,
       0.8817224172427963
      ],
      "fitted_gap": [
       -0.025459999999999993,
       0.058929999999999996,
       0.058929999999999996,
       0.0637
      ]
     }
    },
    "vgg_fd_isotonic": {
     "cifar10": {
      "d": [
       -1.6837863547428957,
       0.24776727118698408,
       0.5277472403349504,
       0.9082718432209619
      ],
      "fitted_gap": [
       -0.003773333333333332,
       -0.003773333333333332,
       -0.003773333333333332,
       0.001959999999999984
      ]
     },
     "cifar100": {
      "d": [
       -1.5406418856247068,
       -0.1060605376255031,
       0.4632405986454712,
       1.1834618246047375
      ],
      "fitted_gap": [
       0.004499999999999993,
       0.07934,
       0.08887999999999999,
       0.08887999999999999
      ]
     },
     "supercifar100": {
      "d": [
       -1.5309506222466422,
       -0.10307115953828566,
       0.4229595385617346,
       1.2110622432231894
      ],
      "fitted_gap": [
       -0.009080000000000013,
       -0.009080000000000013,
       -0.009080000000000013,
       -0.00606000000000001
      ]
     },
     "tinyimagenet": {
      "d": [
       -1.6404351537683408,
       0.0745095730469534,
       0.5862912779704899,
       0.9796343027508974
      ],
      "fitted_gap": [
       -0.025459999999999993,
       0.06052,
       0.06052,
       0.06052
      ]
     }
    }
   },
   "source_shift_mean": {
    "cifar10|fashionmnist_new": -0.007220000000000027,
    "cifar10|kmnist_new": -0.009240000000000003,
    "cifar10|mnist_new": 0.001959999999999984,
    "cifar10|stl10_new": 0.005140000000000033,
    "cifar100|fashionmnist_new": 0.07934,
    "cifar100|kmnist_new": 0.09574,
    "cifar100|mnist_new": 0.08201999999999998,
    "cifar100|stl10_new": 0.004499999999999993,
    "supercifar100|fashionmnist_new": -0.0021800000000000265,
    "supercifar100|kmnist_new": -0.022659999999999993,
    "supercifar100|mnist_new": -0.00606000000000001,
    "supercifar100|stl10_new": -0.0024000000000000245,
    "tinyimagenet|fashionmnist_new": 0.09305999999999999,
    "tinyimagenet|kmnist_new": 0.0637,
    "tinyimagenet|mnist_new": 0.0248,
    "tinyimagenet|stl10_new": -0.025459999999999993
   },
   "source_majority": {
    "cifar10": -1.0,
    "cifar100": 1.0,
    "supercifar100": -1.0,
    "tinyimagenet": 1.0
   },
   "rule_discrepancies_with_protocol": [
    "geometry percentile g_pct computed per source on the full panel before the CE subset; not recomputed within CV folds or after target deletions",
    "majority baseline selects material cells by |dG| >= 0.01 (AUGRC gap) while the target metric is dA (AUROC gap); fallbacks: non-zero-target majority, then global, then Energy"
   ]
  },
  "aug_view": {
   "folds": {
    "rule": "leave-one-VGG-checkpoint-out",
    "fold_ids": [
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "cifar100_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "cifar10_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "supercifar_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run1_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run2_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run3_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run4_rew2.2",
     "tiny-imagenet-200_paper_sweep__confidnet_bbvgg13_do0_run5_rew2.2"
    ]
   },
   "indicator_sources": [
    "cifar100",
    "supercifar100",
    "tinyimagenet"
   ],
   "reference_source": "cifar10",
   "ridge": {
    "vgg_geometry_severity_ridge": {
     "features": [
      "dK",
      "g_pct",
      "dK*g_pct"
     ],
     "lambda": 10.0,
     "cv_losses": {
      "0.0001": 0.001230544602407013,
      "0.001": 0.001230541718425619,
      "0.01": 0.0012305129070332606,
      "0.1": 0.001230227605741492,
      "1.0": 0.001227629249961061,
      "10.0": 0.0012143770180766999
     },
     "scaler_mean": [
      -3.219646771412954e-16,
      0.5,
      -1.4988010832439614e-16
     ],
     "scaler_sd": [
      0.9999999967294846,
      0.282842712474619,
      0.5744562627750348
     ],
     "beta": {
      "intercept": -0.003864999999999998,
      "source_indicators": {
       "cifar100": 0.06541000000000001,
       "supercifar100": -0.004740000000000001,
       "tinyimagenet": 0.04275499999999999
      },
      "standardized_features": {
       "dK": 0.005665734470721686,
       "g_pct": 0.0030099178652507348,
       "dK*g_pct": 0.008512764681313222
      }
     }
    },
    "vgg_matched_scalar_ridge": {
     "features": [
      "logC",
      "logD",
      "logNC1",
      "s_dict",
      "theta_deg",
      "logit",
      "eta",
      "log_gamma",
      "a",
      "log_rho"
     ],
     "lambda": 0.0001,
     "cv_losses": {
      "0.0001": 0.0006202676805690363,
      "0.001": 0.0006210132569474233,
      "0.01": 0.0006238369262680396,
      "0.1": 0.0006302314737357216,
      "1.0": 0.0006635855569136133,
      "10.0": 0.0009490327833837318
     },
     "scaler_mean": [
      3.787627906174154,
      6.584898215319481,
      -1.2050788623691495,
      12.562711447473122,
      14.93151554636933,
      11.521277332016641,
      0.04745022084978906,
      -0.8153532555202488,
      0.5396389861162767,
      0.17456784970566042
     ],
     "scaler_sd": [
      1.2110819283366436,
      0.6002830669264719,
      0.8407867986255019,
      5.4975153275728745,
      4.914632480013557,
      1.818018653592208,
      0.02597744767569514,
      0.49079608899728405,
      0.1159745217406698,
      0.3108715637363166
     ],
     "beta": {
      "intercept": -0.6457286897352995,
      "source_indicators": {
       "cifar100": 0.8766638881071964,
       "supercifar100": 0.34750333232735825,
       "tinyimagenet": 1.4467125385066428
      },
      "standardized_features": {
       "logC": -6.306754919397801e-11,
       "logD": 1.305505021917622e-10,
       "logNC1": -0.19663870746480824,
       "s_dict": -0.375708808018445,
       "theta_deg": -0.006816549924698918,
       "logit": -0.06244493358795604,
       "eta": 0.053604456626565974,
       "log_gamma": 0.06858829263882309,
       "a": -0.004557754047281498,
       "log_rho": 0.0994270202176655
      }
     }
    },
    "vgg_no_target_batch_ridge": {
     "features": [
      "logC",
      "logD",
      "logNC1",
      "s_dict",
      "theta_deg",
      "logit",
      "eta"
     ],
     "lambda": 10.0,
     "cv_losses": {
      "0.0001": 0.001429376755753756,
      "0.001": 0.0014141538274670407,
      "0.01": 0.0013910020809534632,
      "0.1": 0.0013843656088782558,
      "1.0": 0.0013833599157311235,
      "10.0": 0.0013822624997759164
     },
     "scaler_mean": [
      3.787627906174154,
      6.584898215319481,
      -1.2050788623691495,
      12.562711447473122,
      14.93151554636933,
      11.521277332016641,
      0.04745022084978906
     ],
     "scaler_sd": [
      1.2110819283366436,
      0.6002830669264719,
      0.8407867986255019,
      5.4975153275728745,
      4.914632480013557,
      1.818018653592208,
      0.02597744767569514
     ],
     "beta": {
      "intercept": -0.003978607148333946,
      "source_indicators": {
       "cifar100": 0.06494826258726653,
       "supercifar100": -0.005123621273304778,
       "tinyimagenet": 0.044054787279374036
      },
      "standardized_features": {
       "logC": 1.5493842358621124e-17,
       "logD": 1.5790363757569083e-17,
       "logNC1": 0.0005535346517351747,
       "s_dict": -0.0008470121009349701,
       "theta_deg": -9.323356571793529e-05,
       "logit": -8.122851388775265e-05,
       "eta": -0.00018481191065951544
      }
     }
    }
   },
   "isotonic": {
    "vgg_kid_isotonic": {
     "cifar10": {
      "d": [
       -1.570392530375779,
       -0.16014242666029932,
       0.7916262237845381,
       0.9389087332515403
      ],
      "fitted_gap": [
       -0.005640000000000008,
       -0.005640000000000008,
       -0.005640000000000008,
       0.0014599999999999947
      ]
     },
     "cifar100": {
      "d": [
       -1.2706142527915594,
       -0.6853171993881652,
       0.9389413749888746,
       1.016990077190847
      ],
      "fitted_gap": [
       0.0002800000000000136,
       0.07442000000000001,
       0.08574,
       0.08574
      ]
     },
     "supercifar100": {
      "d": [
       -1.2732109525965163,
       -0.6802831009108223,
       0.9131890670898313,
       1.0403049864175054
      ],
      "fitted_gap": [
       -0.01020666666666666,
       -0.01020666666666666,
       -0.01020666666666666,
       -0.0038000000000000256
      ]
     },
     "tinyimagenet": {
      "d": [
       -1.6112361310487624,
       -0.059736111806425136,
       0.7892498256123917,
       0.8817224172427963
      ],
      "fitted_gap": [
       -0.02842000000000002,
       0.05963,
       0.05963,
       0.06472
      ]
     }
    },
    "vgg_fd_isotonic": {
     "cifar10": {
      "d": [
       -1.6837863547428957,
       0.24776727118698408,
       0.5277472403349504,
       0.9082718432209619
      ],
      "fitted_gap": [
       -0.005640000000000008,
       -0.005640000000000008,
       -0.005640000000000008,
       0.0014599999999999947
      ]
     },
     "cifar100": {
      "d": [
       -1.5406418856247068,
       -0.1060605376255031,
       0.4632405986454712,
       1.1834618246047375
      ],
      "fitted_gap": [
       0.0002800000000000136,
       0.07442000000000001,
       0.08574,
       0.08574
      ]
     },
     "supercifar100": {
      "d": [
       -1.5309506222466422,
       -0.10307115953828566,
       0.4229595385617346,
       1.2110622432231894
      ],
      "fitted_gap": [
       -0.01020666666666666,
       -0.01020666666666666,
       -0.01020666666666666,
       -0.0038000000000000256
      ]
     },
     "tinyimagenet": {
      "d": [
       -1.6404351537683408,
       0.0745095730469534,
       0.5862912779704899,
       0.9796343027508974
      ],
      "fitted_gap": [
       -0.02842000000000002,
       0.06132666666666667,
       0.06132666666666667,
       0.06132666666666667
      ]
     }
    }
   },
   "source_shift_mean": {
    "cifar10|fashionmnist_new": -0.00576000000000001,
    "cifar10|kmnist_new": -0.005640000000000023,
    "cifar10|mnist_new": 0.0014599999999999947,
    "cifar10|stl10_new": -0.005519999999999992,
    "cifar100|fashionmnist_new": 0.07442000000000001,
    "cifar100|kmnist_new": 0.09268000000000003,
    "cifar100|mnist_new": 0.07879999999999995,
    "cifar100|stl10_new": 0.0002800000000000136,
    "supercifar100|fashionmnist_new": -0.0030799999999999938,
    "supercifar100|kmnist_new": -0.023719999999999984,
    "supercifar100|mnist_new": -0.0038000000000000256,
    "supercifar100|stl10_new": -0.0038200000000000013,
    "tinyimagenet|fashionmnist_new": 0.09433999999999998,
    "tinyimagenet|kmnist_new": 0.06472,
    "tinyimagenet|mnist_new": 0.024920000000000032,
    "tinyimagenet|stl10_new": -0.02842000000000002
   },
   "source_majority": {
    "cifar10": -1.0,
    "cifar100": 1.0,
    "supercifar100": -1.0,
    "tinyimagenet": 1.0
   },
   "rule_discrepancies_with_protocol": [
    "geometry percentile g_pct computed per source on the full panel before the CE subset; not recomputed within CV folds or after target deletions",
    "majority baseline selects material cells by |dG| >= 0.01 (AUGRC gap) while the target metric is dA (AUROC gap); fallbacks: non-zero-target majority, then global, then Energy"
   ]
  }
 },
 "correction_record": {
  "original_readout": {
   "path": "rn18_handoff_replication/outputs/rn18_report.json",
   "sha256": "a42e863875c2984c6320c7866292da2d96759d3fbb6a884134c54ac3846ab380",
   "bytes": 25728,
   "reader_of_record_commit": "c7d1b98"
  },
  "reader_discrepancy": "frozen hash 038cf044... (commit 6ae5b4b) vs reader of record c50c47fa... (commit c7d1b98): the only difference is the JSON serialization of the tertile composition keys; no numerical path changed",
  "rule_changes": [
   "validator before any computation (freeze hashes, inventory, licenses, key set, families, phase-1, axes, denominators)",
   "non-finite comparator prediction = comparator NOT ESTIMABLE (was: tie with probability 0.5)",
   "non-finite required input on a registered panel = endpoint NOT ESTIMABLE (was: cell dropped)",
   "zero or non-finite jackknife/LEVEL standard error = NOT ESTIMABLE (was: LEVEL had no guard)",
   "decisions on full-precision intervals (was: five-decimal rounded)",
   "multipliers from the version-2 qualification license (was: version-1 development values)",
   "evidence-class label on every endpoint; both denominators and the constant-policy identity reported"
  ],
  "conclusions_changed": {
   "/SEL_ce": {
    "readout_of_record": "PRACTICALLY EQUIVALENT TO THE REFERENCE",
    "v2": "NO LICENSE: descriptive"
   },
   "/SEL_ce_augview_comparators_sensitivity": {
    "readout_of_record": "UNRESOLVED",
    "v2": "NO LICENSE: descriptive"
   },
   "/LEVEL_ce": {
    "readout_of_record": "resolved improvement",
    "v2": "NO LICENSE: descriptive"
   },
   "/LEVEL_ce_do0_sensitivity_Nf5": {
    "readout_of_record": "resolved improvement",
    "v2": "NO LICENSE: descriptive"
   }
  },
  "verdicts_compared": 30,
  "verdicts_only_in_v2": [
   "/validation"
  ]
 }
}
```
