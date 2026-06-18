import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon

def create_edm_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 11))

    ax.set_xlim(0, 18)
    ax.set_ylim(-1, 12)
    ax.axis('off')

    plt.title('Ensemble Model Architecture',
              fontsize=22, fontweight='bold', pad=20)

    # ===== Colors =====
    input_color = '#87CEFA'

    model_colors = [
        '#FF69B4',  # Ridge
        '#F0E68C',  # NN
        '#66CDAA',  # KNN
        '#5F9EA0',  # LSTM
        '#FFD39B',  # XGBoost
        '#D8BFD8'   # SVR
    ]

    ensemble_color = '#ADFF2F'
    output_color = '#D8BFD8'
    line_color = '#000000'

    # ===== Box sizes =====
    box_w, box_h = 3.0, 1.2

    # ================= DRAW FUNCTIONS =================

    def draw_rectangle(x, y, text, fc, fontsize=12):
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="square,pad=0.1",
                facecolor=fc,
                edgecolor='black',
                linewidth=1.4
            )
        )
        ax.text(
            x, y, text,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold'
        )

    def draw_box(x, y, text, fc, fontsize=11):
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.25",
                facecolor=fc,
                edgecolor='black',
                linewidth=1.4
            )
        )
        ax.text(
            x, y, text,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold',
            wrap=True
        )

    def draw_ellipse(x, y, text, fc, fontsize=12):
        ax.add_patch(
            Ellipse(
                (x, y),
                width=box_w * 1.2,
                height=box_h * 1.5,
                facecolor=fc,
                edgecolor='black',
                linewidth=1.4
            )
        )

        ax.text(
            x, y, text,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold'
        )

    def draw_snipped_output_box(x, y, text, fc, fontsize=12, snip=0.3):

        w, h = box_w, box_h

        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y - h / 2, y + h / 2

        points = [
            (x0, y1),
            (x1 - snip, y1),
            (x1, y1 - snip),
            (x1, y0),
            (x0 + snip, y0),
            (x0, y0 + snip)
        ]

        poly = Polygon(
            points,
            closed=True,
            facecolor=fc,
            edgecolor='black',
            linewidth=1.4
        )

        ax.add_patch(poly)

        ax.text(
            x, y, text,
            ha='center',
            va='center',
            fontsize=fontsize,
            fontweight='bold'
        )

    # ================= X POSITIONS =================

    x_input = 3
    x_model = 8
    x_ens = 12
    x_out = 16

    # ================= INPUT LAYER =================

    ax.text(
        x_input,
        11.3,
        'Input Layer',
        fontsize=14,
        fontweight='bold',
        ha='center'
    )

    input_params = [
        'Pulse On',
        'Pulse Off',
        'Servo Voltage',
        'Applied Current',
        'Angle of Wire'
    ]

    y_inputs = [9.5, 7.5, 5.5, 3.5, 1.5]

    for y, param in zip(y_inputs, input_params):
        draw_rectangle(x_input, y, param, input_color)

    # ================= INDIVIDUAL MODELS =================

    ax.text(
        x_model,
        11.3,
        'Individual Models',
        fontsize=14,
        fontweight='bold',
        ha='center'
    )

    model_boxes = [
        ('XGBoost', '300 Estimators\nlr=0.05, max_depth=6'),
        ('SVR', 'Kernel=RBF\nC, gamma tuned\nEpsilon insensitive'),
        ('Ridge Regression',
         'Linear model\nL2 regularization\nα tuned by CV'),
        ('Neural Network', '3×ReLU Layers\nAdam, Epochs=200'),
        ('KNN', 'k neighbors tuned\nDistance=Euclidean\nWeighted voting'),
        ('LSTM', 'Sequence learning\nAdam optimizer tuned\nDropout regularization')
    ]

    # Increased spacing so SVR box is fully visible
    y_models = [10.0, 8.2, 6.4, 4.6, 2.8, 1.0]

    for (title, desc), y, color in zip(
            model_boxes,
            y_models,
            model_colors):

        draw_box(
            x_model,
            y,
            f'{title}\n{desc}',
            color,
            fontsize=10
        )

    # ================= ENSEMBLE =================

    ax.text(
        x_ens,
        11.3,
        'Weighted Ensemble',
        fontsize=14,
        fontweight='bold',
        ha='center'
    )

    ensemble_y = 5.8

    draw_ellipse(
        x_ens,
        ensemble_y,
        'Weighted Average \nXGB(0.10), SVR(0.15)\nRidge(0.20),NN(0.15)\nKNN(0.20), LSTM(0.20)',
        ensemble_color,
        fontsize=11
    )

    # ================= OUTPUTS =================

    ax.text(
        x_out,
        11.3,
        'Output Layer',
        fontsize=14,
        fontweight='bold',
        ha='center'
    )

    outputs = [
        ('Surface Undulation\n(µm)', 7.2),
        ('Substance Vaporization Rate\n(mm³/min)', 4.8),
        ('Kerf Clearance\n(mm)', 2.4)
    ]

    for label, y in outputs:
        draw_snipped_output_box(
            x_out,
            y,
            label,
            output_color,
            fontsize=11
        )

    # ================= ARROWS =================

    arrowprops = dict(
        arrowstyle='->',
        color=line_color,
        lw=1.5
    )

    # Input → Models
    for y_in in y_inputs:
        for y_m in y_models:
            ax.annotate(
                '',
                xy=(x_model - 1.7, y_m),
                xytext=(x_input + 1.6, y_in),
                arrowprops=arrowprops
            )

    # Models → Ensemble
    for y_m in y_models:
        ax.annotate(
            '',
            xy=(x_ens - 1.75, ensemble_y),
            xytext=(x_model + 1.75, y_m),
            arrowprops=arrowprops
        )

    # Ensemble → Outputs
    for _, y_out in outputs:
        ax.annotate(
            '',
            xy=(x_out - 1.5, y_out),
            xytext=(x_ens + 1.8, ensemble_y),
            arrowprops=arrowprops
        )

    # ================= FOOTER =================

    ax.text(
        9,
        -0.4,
        'Process Flow: Input → Models → Weighted Ensemble → Output',
        fontsize=14,
        fontweight='bold',
        style='italic',
        ha='center'
    )

    plt.tight_layout()
    plt.show()


# ===== RUN =====
create_edm_architecture_diagram()