from sklearn import tree

# 0 for smooth and 1 for bumpy
features = [[140,0],[150,1],[130,0],[170,1]] 

#1 for apples and 0 for oranges
labels = [1,0,1,0]  

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[100.9,1]]))