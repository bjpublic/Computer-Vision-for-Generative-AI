class BaseConfig():
    def __init__(self, orig=None):
        if orig is not None:
            print("cawet")


def getConfigFromDict(obj, inputDict, defaultConfig):
    if not inputDict:
        for member, value in vars(defaultConfig).items():
            setattr(obj, member, value)
    else:
        for member, value in vars(defaultConfig).items():
            setattr(obj, member, inputDict.get(member, value))


def updateConfig(obj, ref):
    if isinstance(ref, dict):
        for member, value in ref.items():
            setattr(obj, member, value)

    else:

        for member, value in vars(ref).items():
            setattr(obj, member, value)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise AttributeError('Boolean value expected.')


def updateParserWithConfig(parser, defaultConfig):

    for name, key in vars(defaultConfig).items():
        if key is None:
            continue

        if isinstance(key, bool):
            parser.add_argument('--' + name, type=str2bool, dest=name)
        else:
            parser.add_argument('--' + name, type=type(key), dest=name)

    parser.add_argument('--overrides',
                        action='store_true',
                        help= "For more information on attribute parameters, \
                        please have a look at \
                        models/trainer/standard_configurations")
    return parser


def getConfigOverrideFromParser(parsedArgs, defaultConfig):

    output = {}
    for arg, value in parsedArgs.items():
        if value is None:
            continue

        if arg in vars(defaultConfig):
            output[arg] = value

    return output


def getDictFromConfig(obj, referenceConfig, printDefault=True):
    output = {}
    for member, value in vars(referenceConfig).items():
        if hasattr(obj, member):
            output[member] = getattr(obj, member)
        elif printDefault:
            output[member] = value

    return output
