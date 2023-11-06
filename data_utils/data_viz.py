import matplotlib.pyplot as plt



def make_plot(qual_ann: dict, quant_ann: dict, filename: str):
    """
    Input: qual_ann - keys = qual annotation types, val = num agreed
            annotations in db;
            quant_ann - keys = quant ann types, val = num agreed
            annotations in db
            filename - output plot name

    Makes bar chart showing ratio of articles annotated with each frame
        component, saves fig to data_summary directory
    """
    labels = list(qual_ann.keys())
    # labels = labels + list(quant_ann.keys())
    # print(labels)

    total_articles = 199066
    annotated = []
    not_annotated = []

    for type in qual_ann.keys():
        count = sum(qual_ann[type].values())
        annotated.append((count/total_articles) * 100)
        not_annotated.append(total_articles-count)
        
    # for type in quant_ann.keys():      
    #     count = sum(quant_ann[type].values())
    #     annotated.append(count)
    #     not_annotated.append(total_articles-count)
        
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8)
    
    # print(labels)
    # print(annotated)

    ax.bar(labels, np.array(annotated), label="Value Obtained", color="#DD6031")
    ax.bar(labels, np.array(not_annotated), bottom=annotated, label="No Value", color="#739BD1")

    ax.set_title("Current State of Dataset")
    ax.set_ylabel("Number of Articles")
    ax.set_xlabel("Frame Component")
    ax.legend()

    plt.show()
    plt.savefig("data_summary/" + filename, transparent=True)
