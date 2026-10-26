"""Render REPORT.md as a polished, self-contained course PDF."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether,
                                ListFlowable, ListItem, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'REPORT.md'
OUTPUT = ROOT/'output/pdf/mp2_report.pdf'
NAVY = colors.HexColor('#15324A')
BLUE = colors.HexColor('#2375A8')
TEAL = colors.HexColor('#2A9D8F')
GOLD = colors.HexColor('#E9A23B')
RED = colors.HexColor('#C84C4C')
INK = colors.HexColor('#1D2730')
MUTED = colors.HexColor('#60717F')
PALE = colors.HexColor('#EEF5F8')


def register_fonts():
    regular = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    mono = Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont('Body', str(regular)))
        pdfmetrics.registerFont(TTFont('Body-Bold', str(bold)))
        pdfmetrics.registerFont(TTFont('Mono', str(mono)))
        return 'Body', 'Body-Bold', 'Mono'
    return 'Helvetica', 'Helvetica-Bold', 'Courier'


BODY_FONT, BOLD_FONT, MONO_FONT = register_fonts()


def inline(value):
    value = html.escape(value, quote=False)
    value = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'`([^`]+)`', rf'<font name="{MONO_FONT}" size="7.5">\1</font>', value)
    value = re.sub(r'&lt;(https?://[^&]+)&gt;',
                   r'<link href="\1" color="#2375A8">\1</link>', value)
    return value


def styles():
    sample = getSampleStyleSheet()
    return {
        'body': ParagraphStyle('Body', parent=sample['BodyText'], fontName=BODY_FONT,
            fontSize=8.2, leading=10.7, textColor=INK, spaceAfter=5),
        'h1': ParagraphStyle('H1', parent=sample['Heading1'], fontName=BOLD_FONT,
            fontSize=15, leading=18, textColor=NAVY, spaceBefore=7, spaceAfter=6),
        'h2': ParagraphStyle('H2', parent=sample['Heading2'], fontName=BOLD_FONT,
            fontSize=10.5, leading=13, textColor=BLUE, spaceBefore=5, spaceAfter=4,
            keepWithNext=True),
        'small': ParagraphStyle('Small', parent=sample['BodyText'], fontName=BODY_FONT,
            fontSize=7.2, leading=9.2, textColor=MUTED),
        'cover_title': ParagraphStyle('CoverTitle', parent=sample['Title'], fontName=BOLD_FONT,
            fontSize=25, leading=30, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12),
        'cover_sub': ParagraphStyle('CoverSub', parent=sample['BodyText'], fontName=BODY_FONT,
            fontSize=10, leading=14, textColor=MUTED, spaceAfter=8),
        'abstract': ParagraphStyle('Abstract', parent=sample['BodyText'], fontName=BODY_FONT,
            fontSize=9, leading=12.5, textColor=INK, backColor=PALE,
            borderColor=colors.HexColor('#C9DDE7'), borderWidth=.7,
            borderPadding=10, spaceBefore=5, spaceAfter=10),
        'table': ParagraphStyle('Table', parent=sample['BodyText'], fontName=BODY_FONT,
            fontSize=7.1, leading=8.6, textColor=INK),
        'table_head': ParagraphStyle('TableHead', parent=sample['BodyText'], fontName=BOLD_FONT,
            fontSize=7.1, leading=8.6, textColor=colors.white),
    }


ST = styles()


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 18, width, 18, fill=1, stroke=0)
    if doc.page > 1:
        canvas.setFont(BODY_FONT, 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(48, height - 35, 'DASE7506 MP2  |  Compact Pairwise Ensembles')
        canvas.drawRightString(width - 48, 26, f'{doc.page}')
        canvas.setStrokeColor(colors.HexColor('#D7E2E8'))
        canvas.line(48, 36, width - 48, 36)
    canvas.restoreState()


def validation_chart():
    labels = ['Vote', 'Raw', 'Tree', 'Neural', 'Linear', 'Final']
    values = [86.22, 90.85, 92.36, 93.73, 93.86, 94.49]
    drawing = Drawing(470, 155)
    chart = VerticalBarChart()
    chart.x = 38; chart.y = 28; chart.height = 100; chart.width = 400
    chart.data = [values]
    chart.valueAxis.valueMin = 84; chart.valueAxis.valueMax = 96
    chart.valueAxis.valueStep = 2; chart.valueAxis.labelTextFormat = '%d%%'
    chart.valueAxis.labels.fontName = BODY_FONT; chart.valueAxis.labels.fontSize = 6.5
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = BODY_FONT; chart.categoryAxis.labels.fontSize = 7
    chart.bars[0].fillColor = BLUE; chart.bars[0].strokeColor = BLUE
    chart.barWidth = 30; chart.barSpacing = 12
    drawing.add(chart)
    drawing.add(String(238, 143, 'Validation accuracy by method', textAnchor='middle',
                       fontName=BOLD_FONT, fontSize=9, fillColor=NAVY))
    drawing.add(String(238, 4, 'Final improves the compact linear branch by 5/798 questions',
                       textAnchor='middle', fontName=BODY_FONT, fontSize=6.7, fillColor=MUTED))
    return drawing


def difficulty_chart():
    drawing = Drawing(470, 128)
    chart = VerticalBarChart()
    chart.x = 42; chart.y = 25; chart.height = 74; chart.width = 390
    chart.data = [[92.07, 80.56, 75.00], [88.78, 76.98, 76.32]]
    chart.valueAxis.valueMin = 65; chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 5; chart.valueAxis.labelTextFormat = '%d%%'
    chart.valueAxis.labels.fontName = BODY_FONT; chart.valueAxis.labels.fontSize = 6.5
    chart.categoryAxis.categoryNames = ['2 answers', '3 answers', '4 answers']
    chart.categoryAxis.labels.fontName = BODY_FONT; chart.categoryAxis.labels.fontSize = 7
    chart.bars[0].fillColor = TEAL; chart.bars[1].fillColor = GOLD
    chart.bars[0].strokeColor = TEAL; chart.bars[1].strokeColor = GOLD
    chart.barWidth = 19; chart.groupSpacing = 22
    drawing.add(chart)
    drawing.add(String(235, 115, 'Accuracy falls as candidate disagreement increases',
                       textAnchor='middle', fontName=BOLD_FONT, fontSize=9, fillColor=NAVY))
    drawing.add(String(160, 3, 'Validation', fontName=BODY_FONT, fontSize=7, fillColor=TEAL))
    drawing.add(String(245, 3, 'Test', fontName=BODY_FONT, fontSize=7, fillColor=GOLD))
    return drawing


def parse_table(lines):
    rows = []
    for index, line in enumerate(lines):
        if index == 1 and set(line.replace('|', '').replace(':', '').replace('-', '').strip()) == set():
            continue
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        style = ST['table_head'] if not rows else ST['table']
        rows.append([Paragraph(inline(cell), style) for cell in cells])
    columns = len(rows[0])
    if columns == 3:
        widths = [2.55*inch, 1.25*inch, 2.75*inch]
    elif columns == 4:
        widths = [1.45*inch, 1.7*inch, 1.7*inch, 1.7*inch]
    else:
        widths = [6.55*inch/columns] * columns
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), .35, colors.HexColor('#CBD8DF')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F6F9FA')]),
    ]))
    return table


def parse_body(text):
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith('## 1.'))
    lines = lines[start:]
    story = []
    table_count = 0
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1; continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), ST['h1']))
            index += 1; continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), ST['h2']))
            index += 1; continue
        if line.startswith('|'):
            block = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                block.append(lines[index].strip()); index += 1
            story.append(parse_table(block)); story.append(Spacer(1, 7))
            table_count += 1
            if table_count == 1:
                story.append(validation_chart()); story.append(Spacer(1, 5))
            elif table_count == 2:
                story.append(difficulty_chart()); story.append(Spacer(1, 5))
            continue
        if line.startswith('- '):
            items = []
            while index < len(lines) and lines[index].strip().startswith('- '):
                items.append(ListItem(Paragraph(inline(lines[index].strip()[2:]), ST['body']),
                                      leftIndent=10))
                index += 1
            story.append(ListFlowable(items, bulletType='bullet', start='circle',
                                      leftIndent=17, bulletFontName=BODY_FONT,
                                      bulletFontSize=6, spaceAfter=4))
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not any([
            lines[index].strip().startswith('## '),
            lines[index].strip().startswith('### '),
            lines[index].strip().startswith('|'),
            lines[index].strip().startswith('- '),
        ]):
            paragraph.append(lines[index].strip()); index += 1
        story.append(Paragraph(inline(' '.join(paragraph)), ST['body']))
    return story


def main():
    text = SOURCE.read_text()
    abstract = text.split('## Abstract', 1)[1].split('## 1.', 1)[0].strip().replace('\n', ' ')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=letter,
                          leftMargin=48, rightMargin=48, topMargin=47, bottomMargin=46,
                          title='Selecting Reliable LLM Math Solutions with Compact Pairwise Ensembles',
                          author='Independent DASE7506 MP2 submission')
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='body')
    doc.addPageTemplates(PageTemplate(id='main', frames=[frame], onPage=page_decor))

    story = [Spacer(1, 55),
             Paragraph('Selecting Reliable LLM Math Solutions with Compact Pairwise Ensembles',
                       ST['cover_title']),
             Paragraph('DASE7506 Advanced Machine Learning Methods  |  MP2  |  Clean-room submission',
                       ST['cover_sub']), Spacer(1, 14)]
    metrics = [
        [Paragraph('<b>94.4862%</b><br/><font size="7">validation</font>', ST['cover_sub']),
         Paragraph('<b>84.5824%</b><br/><font size="7">test, post-freeze</font>', ST['cover_sub']),
         Paragraph('<b>1.08 s</b><br/><font size="7">test + scoring</font>', ST['cover_sub']),
         Paragraph('<b>1.304 MiB</b><br/><font size="7">inference assets</font>', ST['cover_sub'])],
    ]
    card = Table(metrics, colWidths=[1.62*inch]*4, rowHeights=[.78*inch])
    card.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE), ('BOX', (0, 0), (-1, -1), .8, BLUE),
        ('INNERGRID', (0, 0), (-1, -1), .5, colors.HexColor('#C9DDE7')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.extend([card, Spacer(1, 24), Paragraph('Abstract', ST['h1']),
                  Paragraph(inline(abstract), ST['abstract']), Spacer(1, 18),
                  Paragraph('Frozen protocol', ST['h2']),
                  Paragraph('Train labels only; validation for development; one test evaluation after source and asset hashes were frozen. NumPy-only offline inference. Four CPU threads. No external data, pretrained weights, answer lookup or evaluation-time API.', ST['body']),
                  PageBreak()])
    story.extend(parse_body(text))
    doc.build(story)
    print(OUTPUT)


if __name__ == '__main__':
    main()
