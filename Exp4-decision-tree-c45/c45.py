import math


class Node:
    def __init__(self, isLeaf, label, threshold):
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []


class C45:
    """Decision Tree Classifier using C4.5 algorithm."""

    def __init__(self):
        self.data = []
        self.classes = []
        self.numAttributes = -1
        self.attrValues = {}
        self.attributes = []
        self.tree = None

    def preprocessData(self):
        for index, row in enumerate(self.data):
            for attr_index in range(self.numAttributes):
                if(not self.isAttrDiscrete(self.attributes[attr_index])):
                    self.data[index][attr_index] = float(
                        self.data[index][attr_index])

    def log(self, x):
        if x == 0:
            return 0
        else:
            return math.log(x, 2)

    def getMajClass(self, curData):
        freq = [0]*len(self.classes)
        for row in curData:
            index = self.classes.index(row[-1])
            freq[index] += 1
        maxInd = freq.index(max(freq))
        return self.classes[maxInd]

    def allSameClass(self, data):
        for row in data:
            if row[-1] != data[0][-1]:
                return False
        return data[0][-1]

    def isAttrDiscrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

    def entropy(self, dataSet):
        S = len(dataSet)
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for row in dataSet:
            classIndex = list(self.classes).index(row[-1])
            num_classes[classIndex] += 1
        num_classes = [x/S for x in num_classes]
        ent = 0
        for num in num_classes:
            ent += num*self.log(num)
        return ent*-1

    def gain(self, unionSet, subsets):
        # input : data and disjoint subsets of it
        S = len(unionSet)
        weights = [len(subset)/S for subset in subsets]
        impurityAfterSplit = 0
        for i in range(len(subsets)):
            impurityAfterSplit += weights[i]*self.entropy(subsets[i])

        totalGain = self.entropy(unionSet) - impurityAfterSplit
        return totalGain

    def split_info(self, unionSet, subsets):
        ans = 0
        S = len(unionSet)
        probs = [len(subset)/S for subset in subsets]
        for p in probs:
            ans += p * self.log(p)
        return ans*-1

    def gain_ratio(self, unionSet, subsets):
        return self.gain(unionSet, subsets)/self.split_info(unionSet, subsets)

    def fit(self, X, y, classes, attrValues):
        self.classes = classes
        self.attrValues = attrValues
        self.numAttributes = len(self.attrValues.keys())
        self.attributes = list(self.attrValues.keys())
        self.data = []
        for (idx, row) in enumerate(X):
            new_row = list(row) + [y[idx]]
            self.data.append(new_row)
        self.preprocessData()
        self.tree = self.recursiveGenerateTree(self.data, self.attributes)

    def recursiveGenerateTree(self, curData, curAttributes):
        allSame = self.allSameClass(curData)

        if len(curData) == 0:
            # Fail
            return Node(True, "Fail", None)
        elif allSame is not False:
            # return a node with that class
            return Node(True, allSame, None)
        elif len(curAttributes) == 0:
            # return a node with the majority class
            majClass = self.getMajClass(curData)
            return Node(True, majClass, None)
        else:
            (best, best_threshold, splitted) = self.splitAttribute(
                curData, curAttributes)
            remainingAttributes = curAttributes[:]
            remainingAttributes.remove(best)
            node = Node(False, best, best_threshold)
            node.children = [self.recursiveGenerateTree(
                subset, remainingAttributes) for subset in splitted]
            return node

    def splitAttribute(self, curData, curAttributes):
        splitted = []
        maxEnt = -1*float("inf")
        best_attribute = -1

        # None for discrete attributes, threshold value for continuous attributes
        best_threshold = None
        for attribute in curAttributes:
            indexOfAttribute = self.attributes.index(attribute)
            if self.isAttrDiscrete(attribute):
                # split curData into n-subsets, where n is the number of
                # different values of attribute i. Choose the attribute with
                # the max gain
                valuesForAttribute = self.attrValues[attribute]
                subsets = [[] for a in valuesForAttribute]

                for row in curData:
                    for index in range(len(valuesForAttribute)):
                        if row[-1] == valuesForAttribute[index]:
                            subsets[index].append(row)
                            break

                e = self.gain(curData, subsets)
                if e > maxEnt:
                    maxEnt = e
                    splitted = subsets
                    best_attribute = attribute
                    best_threshold = None
            else:
                # Sort the data according to the column. Then try all possible adjacent pairs.
                # Choose the one that yields maximum gain.
                curData.sort(key=lambda x: x[indexOfAttribute])
                for j in range(0, len(curData) - 1):
                    if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
                        threshold = (curData[j][indexOfAttribute] +
                                     curData[j+1][indexOfAttribute]) / 2
                        less = []
                        greater = []
                        for row in curData:
                            if(row[indexOfAttribute] > threshold):
                                greater.append(row)
                            else:
                                less.append(row)
                        e = self.gain(curData, [less, greater])
                        if e >= maxEnt:
                            splitted = [less, greater]
                            maxEnt = e
                            best_attribute = attribute
                            best_threshold = threshold
        return (best_attribute, best_threshold, splitted)

    def predict(self, X):
        y_pred = [self._walk_down(self.tree, sample) for sample in X]
        return y_pred

    def _walk_down(self, node, sample):
        if node.isLeaf:
            return node.label

        feature_name = node.label
        feature_id = self.attributes.index(feature_name)
        feature_value = sample[feature_id]

        if node.threshold is None:
            feature_value_id = self.attrValues[feature_name].tolist().index(
                feature_value)
            return self._walk_down(node.children[feature_value_id], sample)
        else:
            if(float(feature_value) < node.threshold):
                return self._walk_down(node.children[0], sample)
            else:
                return self._walk_down(node.children[1], sample)

    def printTree(self):
        self.printNode(self.tree)

    def printNode(self, node, indent=""):
        if not node.isLeaf:
            if node.threshold is None:
                # discrete
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(indent + node.label + " = " +
                              self.attributes[index] + " : " + child.label)
                    else:
                        print(indent + node.label + " = " +
                              self.attributes[index] + " : ")
                        self.printNode(child, indent + "	")
            else:
                # numerical
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.isLeaf:
                    print(indent + node.label + " <= " +
                          str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " +
                          str(node.threshold)+" : ")
                    self.printNode(leftChild, indent + "	")

                if rightChild.isLeaf:
                    print(indent + node.label + " > " +
                          str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " +
                          str(node.threshold) + " : ")
                    self.printNode(rightChild, indent + "	")
