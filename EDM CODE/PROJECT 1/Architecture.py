import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon

def create_edm_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')

    plt.title('Ensemble Model Architecture',
              fontsize=22, fontweight='bold', pad=20)

    # ===== Colors =====
    input_color = '#87CEFA'
    model_colors = ['#FF69B4', '#F0E68C', '#66CDAA', '#5F9EA0']
    ensemble_color = '#ADFF2F'
    output_color = '#D8BFD8'
    line_color = '#000000'

    # ===== Box sizes =====
    box_w, box_h = 3.0, 1.2

    # ===== Draw Rectangle (for inputs) =====
    def draw_rectangle(x, y, text, fc, fontsize=12):
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="square,pad=0.1",
                facecolor=fc,
                edgecolor='black',
                linewidth=1.4
            )
        )
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

    # ===== Draw Rounded Box (for models) =====
    def draw_box(x, y, text, fc, fontsize=12):
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.25",
                facecolor=fc,
                edgecolor='black',
                linewidth=1.4
            )
        )
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

    # ===== Draw Ellipse (for Ensemble) =====
    def draw_ellipse(x, y, text, fc, fontsize=12):
        ax.add_patch(
            Ellipse(
                (x, y),
                width=box_w*1.2,
                height=box_h*1.5,
                facecolor=fc,
                edgecolor='black',
                linewidth=1.4
            )
        )
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

    # ===== Snipped Output Box =====
    def draw_snipped_output_box(x, y, text, fc, fontsize=12, snip=0.3):
        w, h = box_w, box_h
        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y - h / 2, y + h / 2

        points = [
            (x0, y1),          # Top-left
            (x1 - snip, y1),   # Top-right start
            (x1, y1 - snip),   # Top-right snip
            (x1, y0),          # Bottom-right
            (x0 + snip, y0),   # Bottom-left start
            (x0, y0 + snip)    # Bottom-left snip
        ]

        poly = Polygon(points, closed=True, facecolor=fc, edgecolor='black', linewidth=1.4)
        ax.add_patch(poly)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # ===== X positions =====
    x_input, x_model, x_ens, x_out = 3.0, 8.0, 12.0, 16.0

    # ==================== INPUT LAYER ====================
    ax.text(x_input, 11.2, 'Input Layer', fontsize=16, fontweight='bold', ha='center')

    input_params = [
        'Pulse on',
        'Pulse off',
        'Servo Voltage ',
        'Applied Current ',
        'Angle of Wire '
    ]
    spacing = 2.0
    total_input_height = len(input_params) * spacing
    start_y_input = (12 - total_input_height) / 2 + total_input_height - box_h
    y_inputs = [start_y_input - i * spacing for i in range(len(input_params))]

    for y, param in zip(y_inputs, input_params):
        draw_rectangle(x_input, y, param, input_color, fontsize=12)

    # ==================== INDIVIDUAL MODELS ====================
    ax.text(x_model, 11.2, 'Individual Models', fontsize=16, fontweight='bold', ha='center')

    model_boxes = [
        ('XGBoost', '300 Estimators\nlr=0.05, max_depth=6'),
        ('SVR','Kernel=RBF\nC, gamma tuned\nEpsilon insensitive'),
        ('Ridge Regression',
         'Linear model\nL2 regularization\nα tuned by CV'),
        ('Neural Network', '3×ReLU Layers\nAdam, Epochs=200')
    ]
    total_model_height = len(model_boxes) * spacing
    start_y_model = (12 - total_model_height) / 2 + total_model_height - box_h
    y_models = [start_y_model - i * spacing for i in range(len(model_boxes))]

    for (title, desc), y, c in zip(model_boxes, y_models, model_colors):
        draw_box(x_model, y, f"{title}\n{desc}", c, fontsize=11)

    # ==================== ENSEMBLE AVERAGING ====================
    ax.text(x_ens, 11.2, 'Weighted Ensemble', fontsize=16, fontweight='bold', ha='center')
    ensemble_y = 6.0
    draw_ellipse(
        x_ens, ensemble_y,
        'Weighted Ensemble\nXGB(0.15), SVR(0.25)\nRR(0.30), NN(0.30)',
        ensemble_color, fontsize=12
    )

    # ==================== OUTPUT LAYER ====================
    ax.text(x_out, 11.2, 'Output Layer', fontsize=16, fontweight='bold', ha='center')

    outputs = [
        ('Surface Roughness\n(µm)', 7.2),
        ('Material Removal Rate\n(mm³/min)', 4.8)
    ]
    for label, y in outputs:
        draw_snipped_output_box(x_out, y, label, output_color, fontsize=12)

    # ==================== ARROWS ====================
    arrowprops = dict(arrowstyle='->', color=line_color, lw=1.8)

    for y_in in y_inputs:
        for y_m in y_models:
            ax.annotate('', xy=(x_model - box_w / 2.1 - 0.25, y_m),
                        xytext=(x_input + box_w / 2.20 + 0.25, y_in),
                        arrowprops=arrowprops)

    for y_m in y_models:
        ax.annotate('', xy=(x_ens - box_w / 2 - 0.25, ensemble_y),
                    xytext=(x_model + box_w / 2 + 0.25, y_m),
                    arrowprops=arrowprops)

    for _, y_out in outputs:
        ax.annotate('', xy=(x_out - box_w / 2.4 - 0.20, y_out),
                    xytext=(x_ens + box_w / 2.0 + 0.30, ensemble_y),
                    arrowprops=arrowprops)

    ax.text(10, 0.7,
            'Process Flow: Input → Models → Weighted Ensemble → Output',
            fontsize=14, style='italic', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.show()

# ==== RUN ====
create_edm_architecture_diagram()
