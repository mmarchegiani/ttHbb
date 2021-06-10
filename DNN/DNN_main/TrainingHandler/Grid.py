def copyAndAppend(base_list, append_list):
    if len(base_list)==0: return list(map(list, zip(*[append_list])))
    updated_list = []
    for appendix in append_list:
        for entry in base_list:
            entry_copy = entry[:]
            entry_copy.append(appendix)
            updated_list.append(entry_copy)
    return updated_list

def unfold(properties):
    property_order = []
    combinations = []
    for key in properties['grid-search'].split(','):
        parameters = (properties[key].split(','))
        property_order.append(key)
        combinations = copyAndAppend(combinations,parameters) 
    return combinations, property_order

def concatenate(properties):
    property_order = []
    combinations = []
    for key, parameters in properties.items():
        property_order.append(key)
        combinations.append(parameters)
    try: combinations = list(map(list, zip(*combinations)))
    except: raise ValueError('Error: all grid properties must have the same lenght with unfold = False option')
    return combinations, property_order
