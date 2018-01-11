import shelve
from matplotlib import pyplot as plt


def show_graph(filepath):
    saver = shelve.open(filepath)
    graph = []
    epochs = saver["epochs"]
    for i in range(epochs):
        try:
            graph.append(saver["accuracy_"+str(i)])
        except:
            pass 
    graph = graph
    print(graph)
    plt.plot(range(len(graph)), graph, 'o-')
    plt.show()


if __name__ == "__main__":
    save_dir = "./"
    save_path ="save1"
    show_graph(save_dir + save_path)