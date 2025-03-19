import datetime

def get_decay_lines(nuclides:list[str])->dict:
    """ Creates dictionary of check source data
    given a list of check source nuclides. """
    all_decay_lines = {'Ba133':{'energy':[80.9979, 276.3989, 302.8508, 356.0129, 383.8485],
                         'intensity':[0.329, 0.0716, 0.1834, 0.6205, 0.0894],
                         'half_life':[10.551*365.25*24*3600],
                         'activity_date':datetime.date(2014, 3, 19),
                         'activity':1 * 3.7e4},
                'Co60':{'energy':[1173.228, 1332.492],
                        'intensity':[0.9985, 0.999826],
                        'half_life':[1925.28*24*3600],
                        'actvity_date':datetime.date(2014, 3, 19),
                        'activity':0.872 * 3.7e4},
                'Na22':{'energy':[511, 1274.537],
                        'intensity':[1.80, 0.9994],
                        'half_life':[2.6018*365.25*24*3600],
                        'actvity_date':datetime.date(2014, 3, 19),
                        'activity': 5 * 3.7e4},
                'Cs137':{'energy':[661.657],
                         'intensity':[0.851],
                         'half_life':[30.08*365.25*24*3600],
                         'actvity_date':datetime.date(2014, 3, 19),
                         'activity':4.66 * 3.7e4},
                'Mn54':{'energy':[834.848],
                        'intensity':[0.99976],
                        'half_life':[312.20*24*3600],
                        'actvity_date':datetime.date(2016, 5, 2),
                        'activity':6.27 * 3.7e4}}
    decay_lines = {}
    for nuclide in nuclides:
        if nuclide in all_decay_lines.keys():
            decay_lines[nuclide] = all_decay_lines[nuclide]
        else:
            raise Warning(f'{nuclide} not yet added to get_decay_lines()')
    return decay_lines