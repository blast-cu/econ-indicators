import os

RULE_DIR = 'models/psl/data/rules/'
SETTINGS = {
    'all_rules': {
        'rule_dir': os.path.join(RULE_DIR, 'all_rules'),
        'learn': True,
        'combinations': []
    },
    'no_inter': {
        'rule_dir': os.path.join(RULE_DIR, 'no_inter'),
        'learn': False,
        'combinations': []
    },
    'inter_article': {
        'rule_dir': os.path.join(RULE_DIR, 'inter_article'),
        'learn': True,
        'combinations': [('ValFrame', 'ValEconRate'),
                         ('ValFrame', 'ValEconChange'),
                         ('ValEconRate', 'ValFrame'),
                         ('ValEconRate', 'ValEconChange'),
                         ('ValEconChange', 'ValFrame'),
                         ('ValEconChange', 'ValEconRate')]
    },
    'excerpt_article': {
        'rule_dir': os.path.join(RULE_DIR, 'excerpt_article'),
        'learn': True,
        'combinations': [['ValSpin', 'ValFrame'],
                         ['ValSpin', 'ValEconRate'],
                         ['ValSpin', 'ValEconChange'],
                         ['ValType', 'ValFrame'],
                         ['ValType', 'ValEconRate'],
                         ['ValType', 'ValEconChange']]
    },
    'neighbors': {
        'rule_dir': os.path.join(RULE_DIR, 'neighbors'),
        'learn': True,
        'combinations': [['ValSpin', 'ValSpin'],
                         ['ValSpin', 'ValType'],
                         ['ValType', 'ValSpin'],
                         ['ValType', 'ValType']]
    },
    'intuition1_excerpt_article': {
        'rule_dir': os.path.join(RULE_DIR, 'intuition1_excerpt_article'),
        'learn': True,
        'combinations': [['ValSpin', 'ValEconRate'],
                         ['ValSpin', 'ValEconChange']]
    },
    'experiment1': {
        'rule_dir': os.path.join(RULE_DIR, 'experiment1'),
        'learn': True
    },
    'experiment2': {
        'rule_dir': os.path.join(RULE_DIR, 'experiment2'),
        'learn': True
    },'neighbors_agreement': {
        'rule_dir': os.path.join(RULE_DIR, 'neighbors_agreement'),
        'learn': True
    },

}


PREDICATE_MAP = {
    'ValFrame': 'frame',
    'ValEconRate': 'econ_rate',
    'ValEconChange': 'econ_change',
    'ValType': 'type',
    'ValSpin': 'spin',
}

VALUE_MAP = {
    'frame': {
        'business': 0,
        'industry': 1,
        'macro': 2,
        'government': 3,
        'other': 4
    },
    'econ_rate': {
        'good': 0,
        'poor': 1,
        'none': 2
    },
    'econ_change': {
        'better': 0,
        'worse': 1,
        'same': 2,
        'none': 3
    },

    'type': {
        'macro': 0,
        'industry': 1,
        'government': 2,
        'personal': 3,
        'business': 4,
        'other': 5
    },
    # },
    # 'macro_type': {}, 
    # 'industry_type': {},
    # 'gov_type': {},
    # 'expenditure_type': {},
    # 'revenue_type': {},
    'spin': {
        'pos': 0,
        'neg': 1,
        'neutral': 2
    }
}