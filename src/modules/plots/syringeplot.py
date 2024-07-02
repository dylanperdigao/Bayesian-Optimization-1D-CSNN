import matplotlib.pyplot as plt
import seaborn as sns
import os

def syringeplot(df, filename="syringeplot.pdf", path=".", export_eps=False, export_png=False, **kwargs):
    """
    Plot a syringe plot, a boxplot with jittered points

    Parameters
    ----------
        df : pd.DataFrame
            list of DataFrames
        filename : str, optional
            name of the file
        path : str, optional
            path to save the plot
        export_eps: bool, optional
            boolean to export eps file
        export_png: bool, optional
            boolean to export png file
        **kwargs:
            see below

    Keyword Arguments
    ----------
        text_scale: float, optional
            scale of the text (default: 1)
        loc: str, optional
            location of the legend (default: upper right)
        title: str, optional
            title of the plot
        x: str, optional
            x column name (default: config)
        y: str, optional
            y column name (default: recall)
        z: str, optional 
            z column name (default: net)
        x_name: str, optional 
            x label (default: Configuration)
        y_name: str, optional
            y label (default: Recall)
        z_name: str, optional
            z label (default: Network)
    """
    x = "config" if not kwargs.get("x") else kwargs.get("x")
    y = "recall" if not kwargs.get("y") else kwargs.get("y")
    z = "net" if not kwargs.get("z") else kwargs.get("z")
    x_name = "Configuration" if not kwargs.get("x_name") else kwargs.get("x_name")
    y_name = "Recall" if not kwargs.get("y_name") else kwargs.get("y_name")
    z_name = "Network" if not kwargs.get("z_name") else kwargs.get("z_name")
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    # violin plot grouped by network, epoch and batch size
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    # set text scale of axes, ticks and labels
    # get current font size
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)

    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.boxplot(data=df, x=x, y=y, hue=z, ax=ax, gap=0.1, boxprops=dict(alpha=0.2), fliersize=1)
    sns.stripplot(data=df, x=x, y=y, hue=z, ax=ax, dodge=True, jitter=True, alpha=1, size=1)
    # only show legend for the first plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], title=z_name, loc=loc)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.title(kwargs.get("title")) if kwargs.get("title") else None
    if export_png:
        plt.savefig(f"{path}/{filename}.{extension}", format=extension, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
