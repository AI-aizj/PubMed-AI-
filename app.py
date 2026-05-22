import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from Bio import Entrez
from openai import OpenAI

# 配置 PubMed API
Entrez.email = "aizjmomo@126.com"  # 您的邮箱

# 模拟的期刊影响因子与分区数据库 (实际开发中请读取您整理好的本地完整 CSV)
MOCK_IF_DB = {
    "N ENGL J MED": (158.5, "1区"),
    "LANCET": (145.1, "1区"),
    "JAMA": (120.7, "1区"),
    "NATURE": (64.8, "1区"),
    "SCIENCE": (56.9, "1区"),
    "CELL": (64.5, "1区"),
    "PLOS ONE": (3.7, "3区"),
    "BIOINFORMATICS": (5.8, "2区")
}

# ==================== 1. PubMed 数据检索模块 ====================
def search_pubmed(keyword, max_results=100, api_key=None):
    if api_key:
        Entrez.api_key = api_key

    # 检索关键词获取 PMID 列表
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]

    if not id_list:
        return pd.DataFrame()

    # 获取详细信息
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    articles = []
    for article in records.get('PubmedArticle', []):
        try:
            medline = article['MedlineCitation']
            pmid = str(medline['PMID'])
            title = medline['Article']['ArticleTitle']

            # 提取年份
            pub_date = medline['Article']['Journal']['JournalIssue']['PubDate']
            year = pub_date.get('Year', 'Unknown')
            if year == 'Unknown' and 'MedlineDate' in pub_date:
                year = pub_date['MedlineDate'][:4]

            # 提取期刊名
            journal = medline['Article']['Journal']['Title'].upper()
            iso_journal = medline['Article']['Journal']['ISOAbbreviation'].upper()

            # 提取摘要
            abstract_text = ""
            if 'Abstract' in medline['Article']:
                abstract_text = " ".join(medline['Article']['Abstract']['AbstractText'])

            # 匹配影响因子和分区
            if_val, quartile = MOCK_IF_DB.get(iso_journal, MOCK_IF_DB.get(journal, (0.0, "未收录")))

            articles.append({
                "PMID": pmid, "Title": title, "Journal": iso_journal,
                "Year": year, "Abstract": abstract_text, "IF": if_val, "Quartile": quartile
            })
        except Exception as e:
            continue

    return pd.DataFrame(articles)


# ==================== 2. AI 综述生成模块 ====================
def generate_ai_review(df):
    # 筛选有摘要的数据
    valid_abstracts = df[df['Abstract'] != ""]['Abstract'].head(10).tolist()  # 取前10篇做示例
    combined_abstracts = "\n\n".join([f"[{i + 1}] {text}" for i, text in enumerate(valid_abstracts)])

    # 直接配置阿里云百炼端点与您的 API Key
    client = OpenAI(
        api_key="sk-",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    prompt = f"""
    你是一名资深的生物医学专家。请基于以下提供的几篇文献摘要，撰写一份大约500字的研究进展综述报告。
    要求：
    1. 必须使用中文。
    2. 分点呈现（如：核心研究方向、主要技术路径、存在的挑战与未来展望）。
    3. 语言精炼，学术性强，结果直接输出，不要带前言和客套话。

    文献摘要内容如下：
    {combined_abstracts}
    """

    # 阿里云百炼上的 DeepSeek 模型的官方模型 ID 为 deepseek-v3
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content


# ==================== 3. Streamlit 网页布局 ====================
st.set_page_config(page_title="PubMed AI 数据分析工作台", layout="wide")
st.title("🔬 PubMed AI 数据可视化与综述工作台")

# 侧边栏配置
st.sidebar.header("🔑 API 密钥配置")
pubmed_key = st.sidebar.text_input("PubMed API Key (可选，留空限制3次/秒)", type="password")

# 核心搜索框
keyword = st.text_input("输入检索关键词 (例如: Cancer Immunotherapy)", "Cancer Immunotherapy")

if st.button("🚀 开始检索与分析", type="primary"):
    with st.spinner("正在从 PubMed 抓取数据并分析，请稍候..."):
        # 1. 检索数据
        df = search_pubmed(keyword, max_results=100, api_key=pubmed_key)

        if df.empty:
            st.error("未能检索到相关文献，请更换关键词。")
        else:
            st.success(f"成功获取 {len(df)} 篇相关文献数据！")

            # 处理年份格式用于绘图
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            df_filtered = df[df['Year'] > 0]

            # ---------- 核心功能 1：统计相关研究数量、年份、分区 ----------
            st.header("📊 基础文献计量统计")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📅 年度发文量趋势")
                year_counts = df_filtered['Year'].value_counts().sort_index().reset_index()
                year_counts.columns = ['Year', 'Count']
                fig_year = px.line(year_counts, x='Year', y='Count', markers=True)
                st.plotly_chart(fig_year, use_container_width=True)

            with col2:
                st.subheader("🏫 期刊分区分布")
                quartile_counts = df['Quartile'].value_counts().reset_index()
                quartile_counts.columns = ['Quartile', 'Count']
                fig_q = px.pie(quartile_counts, values='Count', names='Quartile', hole=0.4)
                st.plotly_chart(fig_q, use_container_width=True)

            # ---------- 核心功能 2：生成词云与方向概括 ----------
            st.header("☁️ 研究热点词云")
            all_titles = " ".join(df['Title'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(all_titles)

            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)

            # ---------- 核心功能 3：影响力排序（5年内影响因子前100） ----------
            st.header("🏆 近5年高影响力文献排行（Top 100）")
            current_year = 2026  # 系统当前年份
            df_5years = df[df['Year'] >= (current_year - 5)]
            df_sorted = df_5years.sort_values(by="IF", ascending=False).head(100)

            # 在网页展示清洗后的表格
            st.dataframe(df_sorted[['PMID', 'Title', 'Journal', 'Year', 'IF', 'Quartile']], use_container_width=True)

            # ---------- 核心功能 4：基于高影响力文献摘要生成综述 ----------
            st.header("📝 AI 领域简单综述报告 (中文)")
            with st.spinner("AI 正在深度阅读文献摘要并撰写综述..."):
                try:
                    # 移除了外界 llm_key 的校验，直接内部跑您的百炼 Key
                    ai_report = generate_ai_review(df_sorted)
                    st.markdown(f"> {ai_report}")
                except Exception as e:
                    st.error(f"AI 报告生成失败: {str(e)}")