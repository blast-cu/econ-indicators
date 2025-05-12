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
        # 'learn': False,
        'learn': True,
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
    'precedes': {
        'rule_dir': os.path.join(RULE_DIR, 'precedes'),
        'learn': True,
        'combinations': [['ValSpin', 'ValSpin'],
                         ['ValSpin', 'ValType'],
                         ['ValType', 'ValSpin'],
                         ['ValType', 'ValType']]
    },
    'best_rules': {
        'rule_dir': os.path.join(RULE_DIR, 'best_rules'),
        'learn': True,
        'combinations': [['ValType', 'ValType']]
    },
    'combos': {
        'rule_dir': os.path.join(RULE_DIR, 'combos'),
        'learn': True
    },
    'best_12-2024' : {
        'rule_dir': os.path.join(RULE_DIR, 'best_12-2024'),
        'learn': True
    },
    'best_2025-05' : {
        'rule_dir': os.path.join(RULE_DIR, 'best_2025-05'),
        'learn': True
    }
}


PREDICATE_MAP = {
    'ValFrame': 'frame',
    'ValEconRate': 'econ_rate',
    'ValEconChange': 'econ_change',
    'ValType': 'type',
    'ValSpin': 'spin',
}