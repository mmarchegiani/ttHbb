from math import pi

######################################## Selection criteria / Inputs for corrections ############################################

parameters = {
  "muons": {
    "type": "mu",
    "leading_pt": {
      '2016' : 25,
      '2017' : 25,
      '2018' : 25,
      },
    "subleading_pt": 15,
    "eta": 2.4,
    "leading_iso": 0.15,
    "subleading_iso": 0.25,
  },

  "electrons" : {
    "type": "el",
    "leading_pt": {
      '2016' : 25,
      '2017' : 25,
      '2018' : 25,
      },
    "subleading_pt": 15,
    "eta": 2.4
  },
  "jets": {
    "type": "jet",
    "dr": 0.4,
    "pt": 30,
    "eta": 2.4,
    "jetId": 2,
    "puId": 2
  },
  "fatjets": {
    "type": "fatjet",
    "dr": 0.8,
    "pt": 300,
    "eta": 2.4,
    "jetId": 2,
    "tau32cut": 0.4,
    "tau21cut": {
        '2016' : 0.35,
        '2017' : 0.45,
        '2018' : 0.45,
        },
    "tau21DDTcut": {
        '2016' : 0.43,
        '2017' : 0.43,
        '2018' : 0.43,
        },
  },
  "W": {
    'min_mass': 65,
    'max_mass': 105
  },
  "met"   : 40,
  'btags' : 0
}

eraDependentParameters = {
    "2016" : {
        "lumi":  36773.0,
        #"lumimask": "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/ReReco/Final/Cert_271036-284044_13TeV_ReReco_07Aug2017_Collisions16_JSON.txt",
        #"lumimask": "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Final/Cert_271036-284044_13TeV_PromptReco_Collisions16_JSON.txt",
        'lumimask': 'data/Cert_271036-284044_13TeV_PromptReco_Collisions16_JSON.txt',
        "triggers" : {
            "ee" : [
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Ele27_WPTight_Gsf",
            ],
            "emu" : [
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Ele27_WPTight_Gsf",
                "HLT_IsoMu24",
                "HLT_IsoTkMu24",
            ],
            "mumu" : [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
                "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
            ],
        },
        "pu_corrections_file" : "data/PileupData_GoldenJSON_Full2016.root",
        "btag_SF_btagDeepFlavB" : 'data/DeepJet_2016LegacySF_V1_TuneCP5.csv',
        "corrections" : [
            "el_triggerSF Ele27_WPTight_Gsf data/TriggerSF_Run2016All_v1.histo.root",
            "el_recoSF EGamma_SF2D data/2016/EGM2D_BtoH_GT20GeV_RecoSF_Legacy2016.histo.root", ###FIXME for low Et
            "el_idSF EGamma_SF2D data/2016/2016LegacyReReco_ElectronTight_Fall17V2.histo.root",
            "mu_triggerSF IsoMu24_OR_IsoTkMu24_PtEtaBins/pt_abseta_ratio data/2016/EfficienciesAndSF_RunBtoF.histo.root",
            "mu_isoSF NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt data/2016/RunBCDEF_SF_ISO.histo.root", ###FIXME for different runs
            "mu_idSF NUM_TightID_DEN_genTracks_eta_pt data/2016/RunBCDEF_SF_ID.histo.root", ###FIXME for different runs
        ],
        "btagging_algorithm"  : "btagDeepFlavB",#"btagDeepB",
        "btagging_WP"         : 0.3093,
        "bbtagging_algorithm" : "btagDDBvL",#"deepTagMD_HbbvsQCD", #"deepTagMD_bbvsLight",
        "bbtagging_WP"        : 0.86,
        'bbtagSF_DDBvL_M1_loPt'      : 0.86,
        'bbtagSF_DDBvL_M1_loPt_up'   : 0.97,
        'bbtagSF_DDBvL_M1_loPt_down' : 0.82,
        'bbtagSF_DDBvL_M1_hiPt'      : 0.86,
        'bbtagSF_DDBvL_M1_hiPt_up'   : 0.97,
        'bbtagSF_DDBvL_M1_hiPt_down' : 0.82,
    },
    "2017" : {
        "lumi":  41529.0,
        #"lumimask": "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Final/Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt",
        "lumimask": "data/Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt",
        "triggers" : {
            "ee" : [
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Ele32_WPTight_Gsf",
            ],
            "emu" : [
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Ele32_WPTight_Gsf",
                "HLT_IsoMu24_eta2p1",
                "HLT_IsoMu27",
            ],
            "mumu" : [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
                "HLT_IsoMu24_eta2p1",
                "HLT_IsoMu27",
            ],
        },
        "pu_corrections_file" : "data/pileup_Cert_294927-306462_13TeV_PromptReco_Collisions17_withVar.root",
        #"btag_SF_btagDeepFlavB" : "./data/sfs_deepjet_2017_19-11-11.btag.csv",
        "btag_SF_btagDeepFlavB" : 'data/DeepFlavour_2017_94XSF_V4_B_F.csv',
        "corrections" : [
            "el_triggerSF SFs_ele_pt_ele_sceta_ele28_ht150_OR_ele35_2017BCDEF ./data/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb_Data_MC_v5.0.histo.root",
            "el_recoSF EGamma_SF2D data/2017/egammaEffi_EGM2D_runBCDEF_passingRECO.histo.root", ###FIXME for low Et
            "el_idSF EGamma_SF2D data/2017/2017_ElectronTight.histo.root",
            "mu_triggerSF IsoMu27_PtEtaBins/pt_abseta_ratio ./data/2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.histo.root",
            "mu_isoSF NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta data/2017/RunBCDEF_SF_ISO_syst.histo.root",
            "mu_idSF NUM_TightID_DEN_genTracks_pt_abseta data/2017/RunBCDEF_SF_ID_syst.histo.root",
        ],
        "btagging_algorithm"  : "btagDeepFlavB",#"btagDeepB",
        "btagging_WP"         : 0.3033, # 0.4941, # medium working point for btagDeepB
        "bbtagging_algorithm" :  "btagDDBvL",#"deepTagMD_HbbvsQCD", #"deepTagMD_bbvsLight",
        "bbtagging_WP"        : 0.86,## https://indico.cern.ch/event/853828/contributions/3723593/attachments/1977626/3292045/lg-btv-deepak8v2-sf-20200127.pdf
        'bbtagSF_DDBvL_M1_loPt'      : 0.82,
        'bbtagSF_DDBvL_M1_loPt_up'   : 0.86,
        'bbtagSF_DDBvL_M1_loPt_down' : 0.77,
        'bbtagSF_DDBvL_M1_hiPt'      : 0.77,
        'bbtagSF_DDBvL_M1_hiPt_up'   : 0.83,
        'bbtagSF_DDBvL_M1_hiPt_down' : 0.67,
    },
    "2018" : {
        "lumi":  58830.0,
        "lumimask": "data/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt",
        #"lumimask": "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PromptReco/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt",
        "triggers" : {
            "ee" : [
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Ele32_WPTight_Gsf",
            ],
            "emu" : [
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "HLT_Ele32_WPTight_Gsf",
                "HLT_IsoMu24",
            ],
            "mumu" : [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
                "HLT_IsoMu24",
            ],
        },
        "pu_corrections_file" : "data/PileupHistogram-goldenJSON-13tev-2018-100bins_withVar.root",
        "btag_SF_btagDeepFlavB" : 'data/DeepJet_2018_102XSF_V2.csv',
        "corrections" : [
            "el_triggerSF ele28_ht150_OR_ele32_ele_pt_ele_sceta data/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb_Legacy2018_v1.histo.root",
            "el_recoSF EGamma_SF2D data/2018/egammaEffi_EGM2D_updatedAll.histo.root",
            "el_idSF EGamma_SF2D data/2018/2018_ElectronTight.histo.root",
            "mu_triggerSF IsoMu24_PtEtaBins/pt_abseta_ratio data/2018/EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.histo.root",
            "mu_isoSF NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta data/2018/RunABCD_SF_ISO.histo.root",
            "mu_idSF NUM_TightID_DEN_TrackerMuons_pt_abseta data/2018/RunABCD_SF_ID.histo.root",
        ],
        "btagging_algorithm"  : "btagDeepFlavB",#"btagDeepB",
        "btagging_WP"         : 0.2770, # medium working point for btagDeepB
        "bbtagging_algorithm" :  "btagDDBvL",#"deepTagMD_HbbvsQCD", #"deepTagMD_bbvsLight",
        "bbtagging_WP"        : 0.86,
        'bbtagSF_DDBvL_M1_loPt'      : 0.81,
        'bbtagSF_DDBvL_M1_loPt_up'   : 0.88,
        'bbtagSF_DDBvL_M1_loPt_down' : 0.76,
        'bbtagSF_DDBvL_M1_hiPt'      : 0.76,
        'bbtagSF_DDBvL_M1_hiPt_up'   : 0.82,
        'bbtagSF_DDBvL_M1_hiPt_down' : 0.71,
    },

}


#################################### Samples info ##############################################################################

samples_info = {
    "Run_dummy": {
            "XS": 1,
            "ngen_weight": {
                '2016' : 1.,
                '2017' : 1.,
                '2018' : 1.,
                },
            },
    "mc_dummy": {
            "XS": 1,
            "ngen_weight": {
                '2016' : 1.,
                '2017' : 1.,
                '2018' : 1.,
                },
            },
    "ttHTobb": {
            "process": "ttHTobb",
            "XS": 0.2953,
            #"ngen_weight": 2410620.0644499995 #reduced file list
            "ngen_weight": {
                '2016' : 5253482.85,
                '2017' : 4216319.31,
                '2018' : 4996147.01 #11231808.09,
                },
            },
    "THW": {
      "XS":  0.01517,#0.002879,
            "ngen_weight": {
                '2016' : 4989133.86,
                '2017' : 4714331.00,
                '2018' : 14971606.14,
                },
      },
    "ttHToNonbb": {
            "process": "ttHToNonbb",
            "XS": 0.2118,
            #"ngen_weight": 913045.7391360003 #reduced file list
            "ngen_weight": {
                '2016' : 5248991.57,
                '2017' : 3095197.81,
                '2018' : 3963935.78,
                },
            },

    "TTToSemiLeptonic": {
            "XS": 365.4574,
            "ngen_weight": {
                '2016' : 32366940321.33,
                '2017' : 59457024911.66,
                '2018' : 60050384119.23,
                },
            },
    "TTTo2L2Nu": {
            "XS": 88.3419,
            "ngen_weight": {
                '2016' : 4784620999.11,
                '2017' : 648729877.29,
                '2018' : 4622080044.95,
                },
            },
    "TTToHadronic": {
            "XS": 377.9607,
            "ngen_weight": {
                '2016' : 21500086465.24,
                '2017' : 61932449366.28,
                '2018' : 62639466237.,
                },
            },
    "TTZToQQ": {
      "XS": 0.6012,# 0.5297,
            "ngen_weight": {
                '2016' : 396340.93,
                '2017' : 4564905.23,
                '2018' : 4534202.12,
                },
      },
    "ST_s-channel_4f_leptonDecays": {
      "process": "ST_s-channel",
      "XS": 3.36,#3.702224,
      #"ngen_weight": 24856809.513425056 #reduced file list
            "ngen_weight": {
                '2016' : 36768937.25,
                '2017' : 37052021.59,
                '2018' : 74634736.73,
                },
      },
    "ST_tW_antitop": {
      "process": "ST_tW_antitop",
      "XS": 35.85,
      #"ngen_weight": 182291193.36093727 #reduced file list
            "ngen_weight": {
                '2016' : 174109580.67,
                '2017' : 279005351.85,
                '2018' : 266470421.96,
                },
      },
    "ST_tW_top": {
      "process": "ST_tW_top",
      "XS": 35.85,
      #"ngen_weight": 241590614.9098064 #reduced file list
            "ngen_weight": {
                '2016' : 173908712.95,
                '2017' : 272081073.53,
                '2018' : 334874722.20,
                },
      },
    "ST_t-channel_antitop": {
      "process": "ST_t-channel_antitop",
      "XS": 80.95,
            "ngen_weight": {
                '2016' : 17771478.65,
                '2017' : 64689262.55,
                '2018' : 5125996535.38,
                },
      },
    "ST_t-channel_top": {
      "process": "ST_t-channel_top",
      "XS": 136.02,
            "ngen_weight": {
                '2016' : 67975483.38,
                '2017' : 5982064.0,
                '2018' : 16603455266.97,
                },
      },
    "TTGJets": {
      "XS": 3.697,
            "ngen_weight": {
                '2016' : 67622406.44,
                '2017' : 62364926.69,
                '2018' : 33778755.65,
                },
      },
    "TTWJetsToQQ": {
      "XS": 0.3708, #0.4062,
            "ngen_weight": {
                '2016' : 569424.14,
                '2017' : 560315.13,
                '2018' : 580758.84,
                },
      },
#    "WJetsToLNu": {
#      "XS": 61526.7,
#            "ngen_weight": {
#                '2016' : 57402435.0,
#                '2017' : 107612500.0,
#                '2018' : 70389866.80,
#                },
#      },
    "WJetsToLNu_HT-200To400": {
      "XS": 409.3,
            "ngen_weight": {
                '2016' : 4963240.0,
                '2017' : 21192211.0,
                '2018' : 25415129.21,
                },
      },
    "WJetsToLNu_HT-400To600": {
      "XS": 57.91,
            "ngen_weight": {
                '2016' : 1963464.0,
                '2017' : 14250114.0,
                '2018' : 5913597.774,
                },
      },
    "WJetsToLNu_HT-600To800": {
      "XS": 12.93,
            "ngen_weight": {
                '2016' : 3779141.0,
                '2017' : 21582309.0,
                '2018' : 19690762.31,
                },
      },
    "WJetsToLNu_HT-800To1200": {
      "XS": 5.395,
            "ngen_weight": {
                '2016' : 1544513.0,
                '2017' : 20272990.0,
                '2018' : 8357921.27,
                },
      },
    "WJetsToLNu_HT-1200To2500": {
      "XS": 1.081,
            "ngen_weight": {
                '2016' : 244532.0,
                '2017' : 19991892.0,
                '2018' : 7567071.14,
                },
      },
    "WJetsToLNu_HT-2500ToInf": {
      "XS": 0.008060,
            "ngen_weight": {
                '2016' : 253561.0,
                '2017' : 20629585.0,
                '2018' : 3189396.07,
                },
      },
    'TTZToLLNuNu' : {
            'ngen_weight' : {
                '2016' : 502198.8,
                '2017' : 1838397.29,
                '2018' : 3226764.90,
                },
        'XS' : .2432,       #### 2.432e-01 +- 3.054e-04 pb
    },
	'TTWJetsToLNu' : {
            'ngen_weight' : {
		'2016' : 731877.67,
		'2017' : 1654929.98,
		'2018' : 1690091.78,
                },
            'XS' : 0.2181,           #### 2.181e-01 +- 3.838e-04 pb
	},
	'DYJetsToLL' : {
            'ngen_weight' : {
		'2016' : 1897141621849.11,
		'2017' : 3251533677930.31,
		'2018' : 17799598587.56,
                },
            'XS' : 6549.,               #### 6.549e+03 +- 1.334e+01 pb
	},
#	'QCD_HT300to500' : {
#            'ngen_weight' : {
#		'2016' : 1,
#		'2017' : 59459614.0,
#		'2018' : 1,
#                },
#            'XS' : 323600.,           #### 3.236e+05 +- 3.074e+02 pb
#	},
	'QCD_HT500to700' : {
            'ngen_weight' : {
                '2016' : 18560541.0,
		'2017' : 55945526.0,
		'2018' : 55046821.83,
                },
            'XS' : 29990.,               #### 2.999e+04 +- 2.871e+01 pb
	},
	'QCD_HT700to1000' : {
            'ngen_weight' : {
                '2016' : 15629253.0,
		'2017' : 47424646.0,
		'2018' : 48028221.62,
                },
            'XS' : 6351.,                   #### 6.351e+03 +- 6.107e+00 pb
	},
	'QCD_HT1000to1500' : {
            'ngen_weight' : {
                '2016' : 4850746.0,
		'2017' : 16485296.0,
		'2018' : 15403521.91,
                },
            'XS' : 1094.,               #### 1.094e+03 +- 1.056e+00 pb
	},
	'QCD_HT1500to2000' : {
            'ngen_weight' : {
                '2016' : 3970819.0,
		'2017' : 11508604.0,
		'2018' : 10883854.57,
                },
            'XS' : 101.0,                   ### 1.010e+02 +- 1.514e+00 pb
	},
	'QCD_HT2000toInf' : {
            'ngen_weight' : {
                '2016' : 1991645.0,
		'2017' : 5825566.0,
		'2018' : 5412264.53,
                },
            'XS' : 20.23,                       ### 2.023e+01 +- 1.978e-02 pb
	},
    "WW": {
      "XS": 118.7,
            "ngen_weight": {
                '2016' : 6988278.14,
                '2017' : 7765891.02,
                '2018' : 7846135.95,
                },
      },
    "WZ": {
      "XS": 65.5443,
            "ngen_weight": {
                '2016' : 2997571.0,
                '2017' : 3928630.0,
                '2018' : 3884167.00,
                },
      },
    "ZZ": {
      "XS": 15.8274,
            "ngen_weight": {
                '2016' : 998034.0,
                '2017' : 1949768.0,
                '2018' : 1978776.75,
                },
      },
	'WJetsToQQ_HT400to600' : {
            "ngen_weight": {
		'2017' : 9412603.0,
		'2018' : 10046074.61,
                },
        'XS' : 314.6,       #### 3.146e+02 +- 2.797e-01 pb
	},
	'WJetsToQQ_HT600to800' : {
            "ngen_weight": {
		'2017' : 8761386.0,
		'2018' : 15246402.33,
                },
        'XS' : 68.60,
	},
	'WJetsToQQ_HT-800toInf' : {
            "ngen_weight": {
		'2017' : 8028207.0,
		'2018' : 14552824.37,
                },
        'XS' : 34.78,
	},
	'ZJetsToQQ_HT400to600' : {
            "ngen_weight": {
		'2017' : 10285185.0,
		'2018' : 16669536.82,
                },
        'XS' : 146.1,           #### 1.461e+02 +- 1.380e-01 pb
	},
	'ZJetsToQQ_HT600to800' : {
            "ngen_weight": {
		'2017' : 8845052.0,
		'2018' : 14600689.07,
                },
        'XS' : 34.01,           #### 3.401e+01 +- 3.225e-02 pb
	},
	'ZJetsToQQ_HT-800toInf' : {
            "ngen_weight": {
		'2017' : 7764804.0,
		'2018' : 10513086.25,
                },
        'XS' : 18.54,            ##### 1.854e+01 +- 1.772e-02 pb
	},
#	'ZJetsToNuNu_HT-200To400' : {
#            "ngen_weight": {
#		'2017' : 21618510.0,
#                },
#        'XS' : 91.83,            #### 9.183e+01 +- 2.437e-01 pb
#	},
#	'ZJetsToNuNu_HT-400To600' : {
#            "ngen_weight": {
#		'2017' : 9094890.0,
#                },
#        'XS' : 13.10,            #### 1.310e+01 +- 1.239e-02 pb
#	},
#	'ZJetsToNuNu_HT-600To800' : {
#            "ngen_weight": {
#		'2017' : 5615921.0,
#                },
#        'XS' : 3.248,            ##### 3.248e+00 +- 3.083e-03 pb
#	},
#	'ZJetsToNuNu_HT-800To1200' : {
#            "ngen_weight": {
#		'2017' : 2041779.0,
#                },
#        'XS' : 1.496,            #####  1.496e+00 +- 1.425e-03 pb
#	},
#	'ZJetsToNuNu_HT-1200To2500' : {
#            "ngen_weight": {
#		'2017' : 336181.0,
#                },
#        'XS' : 0.3425,            #### 3.425e-01 +- 5.603e-04 pb
#	},
#	'ZJetsToNuNu_HT-2500ToInf' : {
#            "ngen_weight": {
#		'2017' : 6446.0,
#                },
#        'XS' : 0.005264,            #### 5.264e-03 +- 8.654e-06 pb
#	},
#    "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8": {
#            "XS": 365.4574,
#            "ngen_weight": {
#                '2016' : 32366940321.33,
#                '2017' : 720253370.0403845,
#                '2018' : 1.,
#                },
#            "nanoAOD" : {
#                '2016' : '',
#                '2017' : '',
#                '2018' : '',
#                },
#            },
#    "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8": {
#            "XS": 88.3419,
#            "ngen_weight": {
#                '2016' : 4746866164.44,
#                '2017' : 283000430.5968169,
#                '2018' : 1.,
#                },
#            "nanoAOD" : {
#                '2016' : '',
#                '2017' : '',
#                '2018' : '',
#                },
#            },
#    "TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8": {
#            "XS": 377.9607,
#            "ngen_weight": {
#                '2016' : 21500086465.24,
#                '2017' : 1647945788.3386502,
#                '2018' : 1.,
#                },
#            "nanoAOD" : {
#                '2016' : '',
#                '2017' : '',
#                '2018' : '',
#                },
#            },
}


############################################################### Histograms ########################################################

histogram_settings = {
    'variables' : {
        'muons_pt'                  : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{\mu}$ [GeV]"},
        'muons_eta'                 : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta^{\mu}$"},
        'goodmuons_pt'              : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{\mu}$ [GeV]"},
        'goodmuons_eta'             : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta^{\mu}$"},
        'electrons_pt'              : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{e}$ [GeV]"},
        'electrons_eta'             : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta^{e}$"},
        'goodelectrons_pt'          : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{e}$ [GeV]"},
        'goodelectrons_eta'         : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta^{e}$"},
        'jets_pt'                   : {'binning' : (0,1500,151), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{j}$ [GeV]"},
        'jets_eta'                  : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta^{j}$"},
        'goodjets_pt'               : {'binning' : (0,1500,151), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{j}$ [GeV]"},
        'goodjets_eta'              : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta^{j}$"},
        'nleps'                     : {'binning' : (0,5,6),      'xlim' : (0,5),    'xlabel' : "$N_{leps}$"},
        'njets'                     : {'binning' : (0,25,26),    'xlim' : (0,25),   'xlabel' : "$N_{jets}$"},
        'ngoodjets'                 : {'binning' : (0,25,26),    'xlim' : (0,25),   'xlabel' : "$N_{goodjets}$"},
        'btags'                     : {'binning' : (0,20,21),    'xlim' : (0,20),   'xlabel' : "$N_{btags}$"},
        'btags_resolved'            : {'binning' : (0,20,21),    'xlim' : (0,20),   'xlabel' : "$N_{btags,res}$"},
        'nfatjets'                  : {'binning' : (0,10,11),    'xlim' : (0,10),   'xlabel' : "$N_{fatjets}$"},
        'charge_sum'                : {'binning' : (-6,+6,13),   'xlim' : (-6,6),   'xlabel' : "lepton charge sum"},
        'met_pt'                    : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{MET}$ [GeV]"},
        'met_phi'                   : {'binning' : (-pi,pi,129), 'xlim' : (-pi,pi), 'xlabel' : "$\phi_{MET}$"},
        'mll'                       : {'binning' : (0,1500,301), 'xlim' : (0,500),  'xlabel' : "$m_{\ell\ell}$ [GeV]"},
        'leading_jet_pt'            : {'binning' : (0,1500,151), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{j,1}$ [GeV]"},
        'leading_jet_eta'           : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta_{j,1}$ [GeV]"},
        'leading_jet_phi'           : {'binning' : (-pi,pi,129), 'xlim' : (-pi,pi), 'xlabel' : "$\phi_{j,1}$"},
        'leading_jet_mass'          : {'binning' : (0,300,301),  'xlim' : (0,300),  'xlabel' : "$m_{j,1}$ [GeV]"},
        'leading_bjet_pt'           : {'binning' : (0,1500,151), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{b,1}$ [GeV]"},
        'leading_bjet_eta'          : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta_{b,1}$ [GeV]"},
        'leading_bjet_phi'          : {'binning' : (-pi,pi,129), 'xlim' : (-pi,pi), 'xlabel' : "$\phi_{b,1}$"},
        'leading_bjet_mass'         : {'binning' : (0,300,301),  'xlim' : (0,300),  'xlabel' : "$m_{b,1}$ [GeV]"},
        'leadAK8JetMass'            : {'binning' : (0,300,301),  'xlim' : (0,300),  'xlabel' : "FatJet $m_{SD}$"},
        'leadAK8JetPt'              : {'binning' : (0,5000,501), 'xlim' : (0,1000), 'xlabel' : "FatJet $p_T$"},
        'leadAK8JetEta'             : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "FatJet $\eta$"},
        'leadAK8JetPhi'             : {'binning' : (-pi,pi,129), 'xlim' : (-pi,pi), 'xlabel' : "FatJet $\phi$"},
        'leadAK8JetRho'             : {'binning' : (-8,0,801),   'xlim' : (-7,0),   'xlabel' : r'FatJet $\rho$'},
        'leadAK8JetHbb'             : {'binning' : (0,1,51),     'xlim' : (0,1),    'xlabel' : r'btagDDBvL'},
        'leadAK8JetTau21'           : {'binning' : (0,1,41),     'xlim' : (0,1),    'xlabel' : r'FatJet $\tau_{21}$'},
        'lepton_plus_pt'            : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{\ell^+}$ [GeV]"},
        'lepton_plus_eta'           : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta_{\ell^+}$"},
        'lepton_plus_phi'           : {'binning' : (-pi,pi,129), 'xlim' : (-pi,pi), 'xlabel' : "$\phi_{\ell^+}$"},
        'lepton_plus_mass'          : {'binning' : (0,300,301),  'xlim' : (0,300),  'xlabel' : "$m_{\ell^+}$ [GeV]"},
        'lepton_minus_pt'           : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{\ell^-}$ [GeV]"},
        'lepton_minus_eta'          : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta_{\ell^-}$"},
        'lepton_minus_phi'          : {'binning' : (-pi,pi,129), 'xlim' : (-pi,pi), 'xlabel' : "$\phi_{\ell^-}$"},
        'lepton_minus_mass'         : {'binning' : (0,300,301),  'xlim' : (0,300),  'xlabel' : "$m_{\ell^-}$ [GeV]"},
        'leading_lepton_pt'         : {'binning' : (0,2000,201), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{\ell,1}$ [GeV]"},
        'leading_lepton_eta'        : {'binning' : (-4,4,81),    'xlim' : (-4,4),   'xlabel' : "$\eta_{\ell,1}$"},
        'leading_lepton_phi'        : {'binning' : (-pi,pi,129), 'xlim' : (-pi,pi), 'xlabel' : "$\phi_{\ell,1}$"},
        'leading_lepton_mass'       : {'binning' : (0,300,301),  'xlim' : (0,300),  'xlabel' : "$m_{\ell,1}$ [GeV]"},
        'ptll'                      : {'binning' : (0,2000,401), 'xlim' : (0,500),  'xlabel' : "$p_{T}^{\ell\ell}$ [GeV]"},
        'mt_ww'                     : {'binning' : (0,2000,401), 'xlim' : (0,500),  'xlabel' : "$m_{T}^{WW}$ [GeV]"},
        'pnu_x'                     : {'binning' : (-1000,1000,201), 'xlim' : (-300,300), 'xlabel' : r'$p_{\nu_x}$ [GeV]'},
        'pnu_y'                     : {'binning' : (-1000,1000,201), 'xlim' : (-300,300), 'xlabel' : r'$p_{\nu_y}$ [GeV]'},
        'pnu_z'                     : {'binning' : (-1000,1000,201), 'xlim' : (-300,300), 'xlabel' : r'$p_{\nu_z}$ [GeV]'},
        'pnubar_x'                  : {'binning' : (-1000,1000,201), 'xlim' : (-300,300), 'xlabel' : r'$p_{\bar{\nu}_x}$ [GeV]'},
        'pnubar_y'                  : {'binning' : (-1000,1000,201), 'xlim' : (-300,300), 'xlabel' : r'$p_{\bar{\nu}_y}$ [GeV]'},
        'pnubar_z'                  : {'binning' : (-1000,1000,201), 'xlim' : (-300,300), 'xlabel' : r'$p_{\bar{\nu}_z}$ [GeV]'},
        'm_w_plus'                  : {'binning' : (0,1500,301), 'xlim' : (0,500),  'xlabel' : "$m_{W^+}$ [GeV]"},
        'm_w_minus'                 : {'binning' : (0,1500,301), 'xlim' : (0,500),  'xlabel' : "$m_{W^-}$ [GeV]"},
        'm_top'                     : {'binning' : (0,2000,401), 'xlim' : (0,500),  'xlabel' : "$m_{t}$ [GeV]"},
        'm_topbar'                  : {'binning' : (0,2000,401), 'xlim' : (0,500),  'xlabel' : r'$m_{\bar{t}}$ [GeV]'},
        'm_tt'                      : {'binning' : (0,3000,301), 'xlim' : (0,2000), 'xlabel' : r'$m_{t\bar{t}}$ [GeV]'},
        'tt_pt'                     : {'binning' : (0,2000,201), 'xlim' : (0,1000), 'xlabel' : r'$p_{T}^{t\bar{t}}$ [GeV]'},
        'top_pt'                    : {'binning' : (0,2000,201), 'xlim' : (0,1000), 'xlabel' : r'$p_{T}^{t}$ [GeV]'},
        'topbar_pt'                 : {'binning' : (0,2000,201), 'xlim' : (0,1000), 'xlabel' : r'$p_{T}^{\bar{t}}$ [GeV]'},
        'deltaRBBbar'               : {'binning' : (0,5,65),     'xlim' : (0,5),    'xlabel' : r'${\Delta}R_{b,\bar{b}}$'},
        'deltaRHiggsTop'            : {'binning' : (0,5,65),     'xlim' : (0,5),    'xlabel' : r'${\Delta}R_{H,t}$'},
        'deltaRHiggsTopbar'         : {'binning' : (0,5,65),     'xlim' : (0,5),    'xlabel' : r'${\Delta}R_{H,\bar{t}}$'},
        'deltaRHiggsTT'             : {'binning' : (0,5,65),     'xlim' : (0,5),    'xlabel' : r'${\Delta}R_{H,t\bar{t}}$'},
        'deltaRTopTopbar'           : {'binning' : (0,5,65),     'xlim' : (0,5),    'xlabel' : r'${\Delta}R_{t,\bar{t}}$'},
        'deltaPhiBBbar'             : {'binning' : (0,pi,65),    'xlim' : (0,pi),   'xlabel' : r'${\Delta}\phi_{b,\bar{b}}$'},
        'deltaPhiHiggsTop'          : {'binning' : (0,pi,65),    'xlim' : (0,pi),   'xlabel' : r'${\Delta}\phi_{H,t}$'},
        'deltaPhiHiggsTopbar'       : {'binning' : (0,pi,65),    'xlim' : (0,pi),   'xlabel' : r'${\Delta}\phi_{H,\bar{t}}$'},
        'deltaPhiHiggsTT'           : {'binning' : (0,pi,65),    'xlim' : (0,pi),   'xlabel' : r'${\Delta}\phi_{H,t\bar{t}}$'},
        'deltaPhiTopTopbar'         : {'binning' : (0,pi,65),    'xlim' : (0,pi),   'xlabel' : r'${\Delta}\phi_{t,\bar{t}}$'},
        'ttHbb_label'               : {'binning' : (0,1,51),     'xlim' : (0,1),    'xlabel' : 'signal'},
        'weights_nominal'           : {'binning' : (0,2,51),     'xlim' : (0,2),    'xlabel' : 'weights (nominal)'},
    },
    'variables2d' : {
        'm_top_vs_pnu_x' : {
            'pnu_x'                     : {'binning' : (0,1000,201), 'xlim' : (0,300), 'xlabel' : r'|$p_{\nu_x}$| [GeV]'},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_met' : {
            'met'                       : {'binning' : (0,2000,201), 'xlim' : (0,300), 'xlabel' : "$p_{T}^{MET}$ [GeV]"},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_leading_lepton_pt' : {
            'leading_lepton_pt'         : {'binning' : (0,2000,201), 'xlim' : (0,500), 'xlabel' : "$p_{T}^{\ell,1}$ [GeV]"},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_leadAK8JetHbb' : {
            'leadAK8JetHbb'             : {'binning' : (0,1,51),     'xlim' : (0,1),   'xlabel' : r'btagDDBvL'},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_btags' : {
            'btags'                     : {'binning' : (0,20,21),    'xlim' : (0,20),  'xlabel' : "$N_{btags}$"},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_leading_bjet_pt' : {
            'leading_bjet_pt'           : {'binning' : (0,1500,151), 'xlim' : (0,500), 'xlabel' : "$p_{T}^{b,1}$ [GeV]"},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_leading_bjet_eta' : {
            'leading_bjet_eta'          : {'binning' : (0,4,41),     'xlim' : (0,4),   'xlabel' : "|$\eta_{b,1}$| [GeV]"},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_m_w_plus' : {
            'm_w_plus'                  : {'binning' : (0,1500,301), 'xlim' : (0,500), 'xlabel' : "$m_{W^+}$ [GeV]"},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_top_vs_m_w_minus' : {
            'm_w_minus'                 : {'binning' : (0,1500,301), 'xlim' : (0,500), 'xlabel' : "$m_{W^-}$ [GeV]"},
            'm_top'                     : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : "$m_{t}$ [GeV]"},
        },
        'm_topbar_vs_m_w_plus' : {
            'm_w_plus'                  : {'binning' : (0,1500,301), 'xlim' : (0,500), 'xlabel' : "$m_{W^+}$ [GeV]"},
            'm_topbar'                  : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : r'$m_{\bar{t}}$ [GeV]'},
        },
        'm_topbar_vs_m_w_minus' : {
            'm_w_minus'                 : {'binning' : (0,1500,301), 'xlim' : (0,500), 'xlabel' : "$m_{W^-}$ [GeV]"},
            'm_topbar'                  : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : r'$m_{\bar{t}}$ [GeV]'},
        },
        'deltaRBBbar_vs_nleps' : {
            'nleps'                     : {'binning' : (0,5,6),      'xlim' : (0,5),    'xlabel' : "$N_{leps}$"},
            'deltaRBBbar'               : {'binning' : (0,5,65),     'ylim' : (0,5),    'ylabel' : r'${\Delta}R_{b,\bar{b}}$'},
        },
        'deltaRBBbar_vs_met' : {
            'met'                       : {'binning' : (0,2000,201), 'xlim' : (0,300),  'xlabel' : "$p_{T}^{MET}$ [GeV]"},
            'deltaRBBbar'               : {'binning' : (0,5,65),     'ylim' : (0,5),    'ylabel' : r'${\Delta}R_{b,\bar{b}}$'},
        },
        'deltaRBBbar_vs_leadAK8JetPt' : {
            'leadAK8JetPt'              : {'binning' : (0,5000,501), 'xlim' : (0,1000), 'xlabel' : "leadAK8JetPt"},
            'deltaRBBbar'               : {'binning' : (0,5,65),     'ylim' : (0,5),    'ylabel' : r'${\Delta}R_{b,\bar{b}}$'},
        },
        'm_w_minus_vs_m_w_plus' : {
            'm_w_plus'                  : {'binning' : (0,1500,301), 'xlim' : (0,500), 'xlabel' : "$m_{W^+}$ [GeV]"},
            'm_w_minus'                 : {'binning' : (0,1500,301), 'ylim' : (0,500), 'ylabel' : "$m_{W^-}$ [GeV]"},
        },
        'm_topbar_vs_m_top' : {
            'm_top'                     : {'binning' : (0,2000,201), 'xlim' : (0,500), 'xlabel' : "$m_{t}$ [GeV]"},
            'm_topbar'                  : {'binning' : (0,2000,201), 'ylim' : (0,500), 'ylabel' : r'$m_{\bar{t}}$ [GeV]'},
        },
    },
    'fill_opts' : {
        'facecolor': "None",
        'edgecolor': ['#ff8000', '#0080ff'],
        'alpha': 1.0
    },
    'error_opts' : {
        #'label': 'Stat. Unc.',
        #'hatch': '///',
        'facecolor': 'none',
        'edgecolor': (0,0,0,.5),
        'linewidth': 0,
        #'elinewidth': 0
    }
}
