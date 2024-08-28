def buildMaskSplit(noiseGShape,
                   noiseGTexture,
                   categoryVectorDim,
                   attribKeysOrder,
                   attribShift,
                   keySplits=None,
                   mixedNoise=False):
    N1 = noiseGShape
    N2 = noiseGTexture

    if not mixedNoise:
        maskShape = [1 for x in range(N1)] + [0 for x in range(N2)]
        maskTexture = [0 for x in range(N1)] + [1 for x in range(N2)]
    else:
        maskShape = [1 for x in range(N1 + N2)]
        maskTexture = [1 for x in range(N1 + N2)]

    if attribKeysOrder is not None:

        C = categoryVectorDim

        if keySplits is not None:
            maskShape = maskShape + [0 for x in range(C)]
            maskTexture = maskTexture + [0 for x in range(C)]

            for key in keySplits["GShape"]:

                index = attribKeysOrder[key]["order"]
                shift = N1 + N2 + attribShift[index]

                for i in range(shift, shift + len(attribKeysOrder[key]["values"])):
                    maskShape[i] = 1

            for key in keySplits["GTexture"]:

                index = attribKeysOrder[key]["order"]
                shift = N1 + N2 + attribShift[index]
                for i in range(shift, shift + len(attribKeysOrder[key]["values"])):
                    maskTexture[i] = 1
        else:

            maskShape = maskShape + [1 for x in range(C)]
            maskTexture = maskTexture + [1 for x in range(C)]

    return maskShape, maskTexture
