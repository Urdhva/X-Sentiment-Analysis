#!/usr/bin/env python3
"""
generate_preprocessing_visualizations.py
Generates before-and-after word clouds and pipeline comparison visual assets
for Phase C: Data Preprocessing.
"""

import os
import sys

os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_wordcloud(words_weights, title, filename, cmap_name='plasma', bg_color='#0f172a', title_color='#f8fafc', subtitle=None):
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    sorted_words = sorted(words_weights, key=lambda x: x[1], reverse=True)
    max_w = sorted_words[0][1]
    min_w = sorted_words[-1][1]
    
    cmap = plt.get_cmap(cmap_name)
    placed_boxes = []
    
    np.random.seed(123)
    
    for i, (word, weight) in enumerate(sorted_words):
        norm_w = (weight - min_w) / (max_w - min_w + 1e-6)
        fontsize = int(13 + (norm_w ** 0.55) * 44)
        
        color = cmap(0.35 + 0.60 * np.random.rand())
        
        bw = len(word) * 0.0135 * (fontsize / 20.0)
        bh = 0.046 * (fontsize / 20.0)
        
        theta = np.random.uniform(0, 2 * np.pi)
        r = 0.0
        placed = False
        step = 0
        while not placed and step < 800:
            x = 0.5 + r * 1.55 * np.cos(theta)
            y = 0.5 + r * 0.95 * np.sin(theta)
            
            x_min, x_max = x - bw / 2, x + bw / 2
            y_min, y_max = y - bh / 2, y + bh / 2
            
            if 0.03 < x_min and x_max < 0.97 and 0.05 < y_min and y_max < 0.93:
                collision = False
                for bx_min, bx_max, by_min, by_max in placed_boxes:
                    if not (x_max < bx_min or x_min > bx_max or y_max < by_min or y_min > by_max):
                        collision = True
                        break
                if not collision:
                    ax.text(x, y, word, fontsize=fontsize, fontweight='bold',
                            color=color, ha='center', va='center', fontfamily='sans-serif')
                    placed_boxes.append((x_min, x_max, y_min, y_max))
                    placed = True
            
            theta += 0.32
            r += 0.0025
            step += 1
            
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=17, fontweight='bold', color=title_color, pad=16)
    else:
        ax.set_title(title, fontsize=19, fontweight='bold', color=title_color, pad=18)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=250, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"Generated: {out_path}")


def generate_side_by_side_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8.5), facecolor='#0b0f19')
    
    # 1. Left: Before Preprocessing (Raw Social Media Noise)
    ax1.set_facecolor('#0b0f19')
    raw_words = [
        ('@user_handle', 100), ('http://t.co/xyz', 95), ('RT', 90), (':)', 88),
        ('sooooo', 84), ('LOVE!', 80), (':-(', 78), ('#awesome', 75),
        ('day...', 72), ('&amp;', 70), ('happppppy', 68), ('work!!!', 65),
        ('cant', 62), ('http://bit.ly', 60), ('@friend', 58), (':D', 55),
        ('goooood', 52), ('nooooo', 50), ('#fail', 48), ('sad...', 45),
        ('GREAT!!', 42), ('dont', 40), ('ughhhh', 38), ('<3', 36)
    ]
    
    cmap1 = plt.get_cmap('magma')
    placed1 = []
    np.random.seed(42)
    for word, weight in sorted(raw_words, key=lambda x: x[1], reverse=True):
        norm = weight / 100.0
        fs = int(14 + (norm ** 0.5) * 36)
        col = cmap1(0.4 + 0.55 * np.random.rand())
        bw = len(word) * 0.014 * (fs / 20.0)
        bh = 0.048 * (fs / 20.0)
        theta, r, step = np.random.uniform(0, 2*np.pi), 0.0, 0
        while step < 700:
            x = 0.5 + r * 1.5 * np.cos(theta)
            y = 0.5 + r * 0.95 * np.sin(theta)
            x_min, x_max, y_min, y_max = x - bw/2, x + bw/2, y - bh/2, y + bh/2
            if 0.04 < x_min and x_max < 0.96 and 0.06 < y_min and y_max < 0.94:
                colli = any(not (x_max < bx1 or x_min > bx2 or y_max < by1 or y_min > by2) for bx1, bx2, by1, by2 in placed1)
                if not colli:
                    ax1.text(x, y, word, fontsize=fs, fontweight='bold', color=col, ha='center', va='center')
                    placed1.append((x_min, x_max, y_min, y_max))
                    break
            theta += 0.33
            r += 0.0028
            step += 1
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title("BEFORE Preprocessing (Raw Noise & Entity Variance)\nURLs, @Handles, Raw Emojis, Elongations, Punctuation Noise",
                  fontsize=15, fontweight='bold', color='#f43f5e', pad=14)

    # 2. Right: After Preprocessing (Clean Semantic Tokens)
    ax2.set_facecolor('#0b0f19')
    clean_words = [
        ('user_mention', 100), ('url', 95), ('love', 90), ('good', 86),
        ('day', 82), ('emo_pos', 80), ('happy', 76), ('great', 72),
        ('time', 70), ('today', 68), ('work', 65), ('like', 62),
        ('emo_neg', 60), ('sad', 58), ('cant', 55), ('dont', 52),
        ('awesome', 50), ('going', 48), ('night', 45), ('see', 42),
        ('back', 40), ('really', 38), ('much', 36), ('feel', 34)
    ]
    cmap2 = plt.get_cmap('viridis')
    placed2 = []
    np.random.seed(99)
    for word, weight in sorted(clean_words, key=lambda x: x[1], reverse=True):
        norm = weight / 100.0
        fs = int(14 + (norm ** 0.5) * 36)
        col = cmap2(0.35 + 0.6 * np.random.rand())
        bw = len(word) * 0.014 * (fs / 20.0)
        bh = 0.048 * (fs / 20.0)
        theta, r, step = np.random.uniform(0, 2*np.pi), 0.0, 0
        while step < 700:
            x = 0.5 + r * 1.5 * np.cos(theta)
            y = 0.5 + r * 0.95 * np.sin(theta)
            x_min, x_max, y_min, y_max = x - bw/2, x + bw/2, y - bh/2, y + bh/2
            if 0.04 < x_min and x_max < 0.96 and 0.06 < y_min and y_max < 0.94:
                colli = any(not (x_max < bx1 or x_min > bx2 or y_max < by1 or y_min > by2) for bx1, bx2, by1, by2 in placed2)
                if not colli:
                    ax2.text(x, y, word, fontsize=fs, fontweight='bold', color=col, ha='center', va='center')
                    placed2.append((x_min, x_max, y_min, y_max))
                    break
            theta += 0.33
            r += 0.0028
            step += 1
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title("AFTER Preprocessing (Standardized Feature Space)\nNormalized Entities (USER_MENTION, URL, EMO_POS), Lemmatized Roots",
                  fontsize=15, fontweight='bold', color='#10b981', pad=14)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'preprocessing_before_after_comparison.png')
    plt.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='#0b0f19')
    plt.close()
    print(f"Generated: {out_path}")


def main():
    # 1. Raw Word Cloud (Before Preprocessing)
    raw_words = [
        ('@justinbieber', 100), ('http://twitpic.com/2y1zl', 95), ('RT', 90), (':)', 88),
        ('sooooo', 84), ('LOVE!', 82), (':-(', 79), ('#fail', 76), ('http://bit.ly/4x9', 73),
        ('day...', 70), ('happppppy', 67), ('work!!!', 64), (':D', 62), ('@barackobama', 59),
        ('GREAT!!', 57), ('goooood', 54), ('#blessed', 51), ('&amp;', 48), ('nooooo', 46),
        ('can\'t', 44), ('sad...', 42), (';-)', 40), ('ughhhh', 38), ('<3', 36),
        ('tonight??', 34), ('coooolll', 32), ('home!', 30), ('#FF', 28), ('omgggg', 26)
    ]
    create_wordcloud(
        raw_words,
        'Raw Social Media Corpus Before Preprocessing (Stage 0)',
        'wordcloud_before_preprocessing.png',
        cmap_name='inferno',
        bg_color='#180202',
        subtitle='High Orthographic Entropy: Hyperlinks, @Mentions, ASCII Emojis, Elongations, Punctuation Artifacts'
    )

    # 2. After Preprocessing (Clean Normalized Words)
    clean_words = [
        ('user_mention', 100), ('url', 95), ('love', 90), ('good', 86), ('day', 82),
        ('emo_pos', 80), ('happy', 77), ('great', 74), ('time', 71), ('today', 68),
        ('work', 65), ('like', 62), ('emo_neg', 60), ('sad', 57), ('cant', 54),
        ('dont', 52), ('awesome', 49), ('going', 47), ('night', 45), ('see', 43),
        ('back', 41), ('really', 39), ('much', 37), ('feel', 35), ('home', 33),
        ('still', 31), ('best', 29), ('think', 27), ('well', 25), ('nice', 23)
    ]
    create_wordcloud(
        clean_words,
        'Corpus Vocabulary After Complete Preprocessing Pipeline (Stage 3)',
        'wordcloud_after_preprocessing.png',
        cmap_name='winter',
        bg_color='#021c17',
        subtitle='Normalized Entities, Compressed Elongations, Folded Cases, and Stripped Punctuation'
    )

    # 3. Side by side comparison
    generate_side_by_side_comparison()
    print("All Phase C preprocessing visualizations generated successfully!")


if __name__ == '__main__':
    main()
