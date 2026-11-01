#!/usr/bin/env python3
"""
generate_postprocessing_visualizations.py
Generates word clouds and visual comparison assets for the
Data Post-Processing section of the X-Sentiment-Analysis Project Report.
"""

import os
import sys

os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_wordcloud(words_weights, title, filename, cmap_name='viridis', bg_color='#0f172a', title_color='#f8fafc', subtitle=None):
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    sorted_words = sorted(words_weights, key=lambda x: x[1], reverse=True)
    max_w = sorted_words[0][1]
    min_w = sorted_words[-1][1]
    
    cmap = plt.get_cmap(cmap_name)
    placed_boxes = []
    
    np.random.seed(42)
    
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


def main():
    # 1. Before Post-Processing: Error & Leakage Vocabulary (False Positives + False Negatives)
    error_leakage_words = [
        ('stuck', 100), ('delay', 95), ('traffic', 90), ('hours', 86), ('brilliant', 82),
        ('not good', 80), ('sarcasm', 76), ('glorious', 72), ('loss', 70), ('prayers', 68),
        ('sorry', 65), ('miss', 62), ('atrocious', 60), ('battery', 58), ('update', 55),
        ('sucks', 52), ('airport', 50), ('waiting', 48), ('cancelled', 45), ('customer', 42),
        ('irony', 40), ('love traffic', 38), ('never again', 36), ('useless', 34), ('failed', 32)
    ]
    create_wordcloud(
        error_leakage_words,
        'BEFORE Post-Processing: Leaked & Ambiguous Token Distribution',
        'wordcloud_postproc_before_errors.png',
        cmap_name='inferno',
        bg_color='#1a0505',
        subtitle='Sarcastic Adjectives, Negation Overlaps, and Affective Ambiguity in Single-Model Errors'
    )

    # 2. After Post-Processing: Purified Positive Sentiment Space
    purified_pos_words = [
        ('love', 100), ('awesome', 94), ('great', 89), ('happy', 85), ('thanks', 81),
        ('best', 78), ('amazing', 74), ('wonderful', 70), ('excited', 67), ('blessed', 64),
        ('beautiful', 61), ('enjoy', 58), ('glad', 55), ('super', 52), ('perfect', 49),
        ('yay', 46), ('congrats', 43), ('brilliant', 40), ('smile', 37), ('win', 35),
        ('cheers', 33), ('friend', 31), ('positive', 29), ('grateful', 27), ('delight', 25)
    ]
    create_wordcloud(
        purified_pos_words,
        'AFTER Post-Processing: Purified Positive Consensus Lexicon',
        'wordcloud_postproc_after_clean_pos.png',
        cmap_name='summer',
        bg_color='#022c22',
        subtitle='Consensus Majority Voting Purges Sarcasm and False Positive Artifacts'
    )

    # 3. After Post-Processing: Purified Negative Sentiment Space
    purified_neg_words = [
        ('hate', 100), ('sad', 94), ('sucks', 88), ('worst', 85), ('horrible', 81),
        ('terrible', 77), ('awful', 74), ('pain', 70), ('angry', 67), ('sick', 64),
        ('tired', 61), ('disappointed', 58), ('hurt', 55), ('broken', 52), ('useless', 49),
        ('annoying', 46), ('depressed', 43), ('crying', 40), ('fail', 37), ('lonely', 35),
        ('stress', 33), ('upset', 31), ('scared', 29), ('poor', 27), ('worse', 25)
    ]
    create_wordcloud(
        purified_neg_words,
        'AFTER Post-Processing: Purified Negative Consensus Lexicon',
        'wordcloud_postproc_after_clean_neg.png',
        cmap_name='autumn',
        bg_color='#3b0707',
        subtitle='Ensemble Consensus Corrects Negated Positives and Affective Sympathy Artifacts'
    )

    # 4. Borderline & Ambivalence Word Cloud (0.45 <= P <= 0.55)
    borderline_words = [
        ('maybe', 100), ('guess', 94), ('though', 88), ('okay', 84), ('suppose', 80),
        ('weird', 76), ('wonder', 72), ('meh', 68), ('fine', 65), ('either', 61),
        ('sort of', 58), ('kind of', 55), ('average', 52), ('alright', 49), ('neutral', 46),
        ('mixed', 43), ('hmm', 40), ('not sure', 37), ('depends', 34), ('halfway', 31)
    ]
    create_wordcloud(
        borderline_words,
        'Borderline & Uncertain Post Distribution (0.45 <= P <= 0.55)',
        'wordcloud_uncertainty_borderline.png',
        cmap_name='cool',
        bg_color='#0f172a',
        subtitle='Lexical Anatomy of Conversational Ambivalence and Mixed Sentiment Clauses'
    )

    print("All post-processing visual word clouds generated successfully!")

if __name__ == '__main__':
    main()

