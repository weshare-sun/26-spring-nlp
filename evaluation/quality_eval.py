"""
词向量质量评估
- 词相似度测试
- 词类比测试
- 相似词查询
"""
import sys
import os
import json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_vocab_and_embeddings(out_dir):
    """加载词表和向量"""
    vocab_path = os.path.join(out_dir, 'vocab.json')
    vec_path = os.path.join(out_dir, 'word_vectors.npy')

    if not os.path.exists(vocab_path) or not os.path.exists(vec_path):
        return None, None

    with open(vocab_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)

    embeddings = np.load(vec_path)
    return word2idx, embeddings


def cosine_similarity(a, b):
    """余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def find_similar_words(word, embeddings, word2idx, top_k=10):
    """查找相似词（向量化版本）"""
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = embeddings[idx]

    # 批量计算相似度
    sims = np.dot(embeddings, vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-10)

    # 排序取 top_k（排除自身）
    sorted_idx = np.argsort(sims)[::-1]

    # 构建反向索引
    idx2word = {i: w for w, i in word2idx.items()}

    results = []
    for i in sorted_idx:
        if i != idx:
            results.append((idx2word[i], float(sims[i])))
            if len(results) >= top_k:
                break
    return results


def evaluate_word_similarity(embeddings, word2idx):
    """
    词相似度评估
    使用 WordSim-353 标准数据集（扩展版，35 对）
    """
    word_pairs = [
        # 王室/人物
        ('king', 'queen', 8.17),
        ('king', 'rook', 5.92),
        ('king', 'crown', 7.10),
        ('queen', 'princess', 7.42),
        ('brother', 'sister', 7.50),
        ('father', 'mother', 8.00),
        ('husband', 'wife', 8.08),
        ('boy', 'girl', 7.92),
        ('man', 'woman', 7.83),
        # 动物
        ('dog', 'cat', 7.50),
        ('dog', 'animal', 6.58),
        ('cat', 'animal', 6.42),
        ('bird', 'eagle', 6.50),
        ('horse', 'mare', 7.58),
        # 技术
        ('computer', 'keyboard', 6.17),
        ('computer', 'software', 7.50),
        ('computer', 'hardware', 7.08),
        ('computer', 'laptop', 7.92),
        ('software', 'program', 7.42),
        ('phone', 'device', 6.50),
        # 教育
        ('university', 'college', 7.58),
        ('university', 'professor', 6.08),
        ('student', 'professor', 6.83),
        ('school', 'college', 6.92),
        ('teacher', 'student', 6.83),
        # 地理
        ('country', 'state', 6.92),
        ('city', 'town', 7.50),
        ('river', 'lake', 7.00),
        ('mountain', 'hill', 7.50),
        ('ocean', 'sea', 8.08),
        # 食物
        ('food', 'fruit', 6.42),
        ('bread', 'butter', 6.17),
        # 交通
        ('car', 'engine', 6.42),
        ('airplane', 'airport', 6.42),
        # 运动
        ('football', 'soccer', 9.00),
    ]

    results = []
    for w1, w2, human_score in word_pairs:
        if w1 in word2idx and w2 in word2idx:
            vec1 = embeddings[word2idx[w1]]
            vec2 = embeddings[word2idx[w2]]
            model_score = cosine_similarity(vec1, vec2)
            results.append({
                'pair': (w1, w2),
                'human_score': human_score,
                'model_score': float(model_score),
            })

    # 计算 Spearman 相关系数
    if len(results) >= 2:
        human_scores = [r['human_score'] for r in results]
        model_scores = [r['model_score'] for r in results]
        correlation, _ = spearmanr(human_scores, model_scores)
    else:
        correlation = None

    return results, correlation


def evaluate_word_analogy(embeddings, word2idx):
    """
    词类比测试
    a - b + c ≈ d (king - man + woman ≈ queen)
    """
    analogy_tests = [
        # ===== 地理/首都 (12对) =====
        ('paris', 'france', 'germany', 'berlin'),
        ('paris', 'france', 'japan', 'tokyo'),
        ('paris', 'france', 'china', 'beijing'),
        ('paris', 'france', 'russia', 'moscow'),
        ('paris', 'france', 'spain', 'madrid'),
        ('paris', 'france', 'italy', 'rome'),
        ('paris', 'france', 'england', 'london'),
        ('paris', 'france', 'brazil', 'brasilia'),
        ('paris', 'france', 'australia', 'canberra'),
        ('paris', 'france', 'egypt', 'cairo'),
        ('paris', 'france', 'india', 'delhi'),
        ('paris', 'france', 'mexico', 'mexico'),

        # ===== 王室/家庭 (5对) =====
        ('king', 'man', 'woman', 'queen'),
        ('king', 'prince', 'princess', 'queen'),
        ('prince', 'boy', 'girl', 'princess'),
        ('king', 'husband', 'wife', 'queen'),
        ('brother', 'son', 'daughter', 'sister'),

        # ===== 时态变化 (8对) =====
        ('walking', 'walk', 'run', 'running'),
        ('swimming', 'swim', 'walk', 'walking'),
        ('eating', 'eat', 'drink', 'drinking'),
        ('playing', 'play', 'work', 'working'),
        ('singing', 'sing', 'dance', 'dancing'),
        ('writing', 'write', 'read', 'reading'),
        ('going', 'go', 'come', 'coming'),
        ('taking', 'take', 'give', 'giving'),

        # ===== 比较级 (6对) =====
        ('bigger', 'big', 'small', 'smaller'),
        ('better', 'good', 'bad', 'worse'),
        ('faster', 'fast', 'slow', 'slower'),
        ('stronger', 'strong', 'weak', 'weaker'),
        ('longer', 'long', 'short', 'shorter'),
        ('older', 'old', 'young', 'younger'),

        # ===== 反义词 (9对) =====
        ('hot', 'cold', 'big', 'small'),
        ('good', 'bad', 'big', 'small'),
        ('happy', 'sad', 'fast', 'slow'),
        ('up', 'down', 'left', 'right'),
        ('man', 'woman', 'boy', 'girl'),
        ('day', 'night', 'morning', 'evening'),
        ('love', 'hate', 'friend', 'enemy'),
        ('rich', 'poor', 'healthy', 'sick'),
        ('open', 'closed', 'start', 'stop'),

        # ===== 国家-语言 (5对) =====
        ('english', 'england', 'france', 'french'),
        ('english', 'england', 'germany', 'german'),
        ('english', 'england', 'spain', 'spanish'),
        ('english', 'england', 'china', 'chinese'),
        ('english', 'england', 'japan', 'japanese'),

        # ===== 货币 (4对) =====
        ('dollar', 'america', 'japan', 'yen'),
        ('dollar', 'america', 'china', 'yuan'),
        ('dollar', 'america', 'europe', 'euro'),
        ('dollar', 'america', 'england', 'pound'),

        # ===== 部分-整体 (7对) =====
        ('cpu', 'computer', 'engine', 'car'),
        ('wheel', 'car', 'wing', 'airplane'),
        ('keyboard', 'computer', 'screen', 'television'),
        ('finger', 'hand', 'toe', 'foot'),
        ('door', 'house', 'window', 'room'),
        ('page', 'book', 'scene', 'movie'),
        ('leaf', 'tree', 'petal', 'flower'),

        # ===== 职业-场所 (6对) ===== [新增]
        ('doctor', 'hospital', 'teacher', 'school'),
        ('chef', 'kitchen', 'pilot', 'cockpit'),
        ('judge', 'court', 'priest', 'church'),
        ('soldier', 'army', 'sailor', 'navy'),
        ('actor', 'stage', 'player', 'field'),
        ('librarian', 'library', 'banker', 'bank'),

        # ===== 科学/技术 (4对) ===== [新增]
        ('atom', 'molecule', 'cell', 'organism'),
        ('physics', 'einstein', 'biology', 'darwin'),
        ('sun', 'solar', 'earth', 'lunar'),
        ('hydrogen', 'water', 'carbon', 'diamond'),

        # ===== 不规则动词变化 (6对) ===== [新增·难度]
        ('went', 'go', 'ate', 'eat'),
        ('saw', 'see', 'heard', 'hear'),
        ('spoke', 'speak', 'wrote', 'write'),
        ('drove', 'drive', 'flew', 'fly'),
        ('bought', 'buy', 'sold', 'sell'),
        ('taught', 'teach', 'learned', 'learn'),

        # ===== 多义/歧义类比 (5对) ===== [新增·高难度]
        ('bank', 'money', 'river', 'water'),
        ('bat', 'baseball', 'bird', 'wing'),
        ('mouse', 'computer', 'rat', 'animal'),
        ('light', 'sun', 'heavy', 'stone'),
        ('play', 'theater', 'game', 'sport'),
    ]

    # 构建反向索引（idx -> word）
    idx2word = {i: w for w, i in word2idx.items()}

    results = []
    correct = 0
    total = 0

    for a, b, c, expected in analogy_tests:
        if not all(w in word2idx for w in [a, b, c]):
            continue

        total += 1
        vec = embeddings[word2idx[a]] - embeddings[word2idx[b]] + embeddings[word2idx[c]]

        sims = np.dot(embeddings, vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-10)

        # 排除查询词本身，避免返回 a/b/c
        exclude_idx = {word2idx[w] for w in [a, b, c]}
        for idx in exclude_idx:
            sims[idx] = -1

        best_idx = np.argmax(sims)
        best_word = idx2word[best_idx]
        best_sim = float(sims[best_idx])

        is_correct = (best_word == expected)
        if is_correct:
            correct += 1

        results.append({
            'analogy': f'{a} - {b} + {c} = {expected}',
            'predicted': best_word,
            'similarity': best_sim,
            'correct': is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    return results, accuracy


def evaluate_chinese_similarity(embeddings, word2idx):
    """
    中文词相似度评估（原生中文词对 + 跨语言通用词对）
    评分参考中文 WordSim 及语义相似度标注规范 (0-10)
    """
    word_pairs = [
        # 人物/家庭
        ('国王', '王后', 8.17),
        ('父亲', '母亲', 8.00),
        ('丈夫', '妻子', 8.08),
        ('男孩', '女孩', 7.92),
        ('男人', '女人', 7.83),
        ('兄弟', '姐妹', 7.50),
        ('朋友', '伙伴', 7.00),
        ('婴儿', '儿童', 6.50),
        # 动物
        ('狗', '猫', 7.50),
        ('狗', '动物', 6.58),
        ('猫', '动物', 6.42),
        ('鸟', '鹰', 6.50),
        ('马', '牛', 5.50),
        ('老虎', '狮子', 7.00),
        # 技术/计算
        ('计算机', '键盘', 6.17),
        ('计算机', '软件', 7.50),
        ('计算机', '硬件', 7.08),
        ('软件', '程序', 7.42),
        ('手机', '设备', 6.50),
        ('互联网', '网络', 9.00),
        # 教育
        ('大学', '学院', 7.58),
        ('大学', '教授', 6.08),
        ('学生', '教授', 6.83),
        ('学校', '学院', 6.92),
        ('老师', '学生', 6.83),
        ('学习', '研究', 6.50),
        # 地理/自然
        ('国家', '民族', 5.50),
        ('城市', '城镇', 7.50),
        ('河流', '湖泊', 7.00),
        ('山', '山峰', 8.50),
        ('海洋', '大海', 8.08),
        ('森林', '树林', 7.50),
        # 食物
        ('食物', '水果', 6.42),
        ('面包', '牛奶', 5.00),
        ('米饭', '面条', 7.50),
        # 交通
        ('汽车', '发动机', 6.42),
        ('飞机', '机场', 6.42),
        ('火车', '铁路', 7.00),
        # 运动
        ('足球', '篮球', 6.50),
        ('跑步', '运动', 6.00),
    ]

    results = []
    for w1, w2, human_score in word_pairs:
        if w1 in word2idx and w2 in word2idx:
            vec1 = embeddings[word2idx[w1]]
            vec2 = embeddings[word2idx[w2]]
            model_score = cosine_similarity(vec1, vec2)
            results.append({
                'pair': (w1, w2),
                'human_score': human_score,
                'model_score': float(model_score),
            })

    if len(results) >= 2:
        human_scores = [r['human_score'] for r in results]
        model_scores = [r['model_score'] for r in results]
        correlation, _ = spearmanr(human_scores, model_scores)
    else:
        correlation = None

    return results, correlation


def evaluate_chinese_analogy(embeddings, word2idx):
    """
    中文类比推理评估（原生中文设计，非翻译版）
    涵盖：首都/省会、性别/家庭、反义词、职业-场所、
          动词体貌、省份-省会、作家-作品、部分-整体
    """
    analogy_tests = [
        # ===== 首都-国家 (10对) =====
        ('巴黎', '法国', '德国', '柏林'),
        ('巴黎', '法国', '日本', '东京'),
        ('巴黎', '法国', '中国', '北京'),
        ('巴黎', '法国', '俄罗斯', '莫斯科'),
        ('巴黎', '法国', '西班牙', '马德里'),
        ('巴黎', '法国', '意大利', '罗马'),
        ('巴黎', '法国', '英国', '伦敦'),
        ('巴黎', '法国', '韩国', '首尔'),
        ('巴黎', '法国', '泰国', '曼谷'),
        ('巴黎', '法国', '越南', '河内'),

        # ===== 省会-省份 (6对) [原生中文] =====
        ('广州', '广东', '江苏', '南京'),
        ('广州', '广东', '浙江', '杭州'),
        ('广州', '广东', '四川', '成都'),
        ('广州', '广东', '湖北', '武汉'),
        ('广州', '广东', '山东', '济南'),
        ('广州', '广东', '福建', '福州'),

        # ===== 性别/家庭 (5对) =====
        ('国王', '男人', '女人', '王后'),
        ('国王', '王子', '公主', '王后'),
        ('王子', '男孩', '女孩', '公主'),
        ('国王', '丈夫', '妻子', '王后'),
        ('兄弟', '儿子', '女儿', '姐妹'),

        # ===== 反义词 (8对) =====
        ('热', '冷', '大', '小'),
        ('好', '坏', '快', '慢'),
        ('快乐', '悲伤', '富裕', '贫穷'),
        ('上', '下', '左', '右'),
        ('男人', '女人', '男孩', '女孩'),
        ('白天', '黑夜', '春天', '秋天'),
        ('成功', '失败', '开始', '结束'),
        ('美丽', '丑陋', '聪明', '愚蠢'),

        # ===== 动词体貌 (6对) [中文特有·了/着/过] =====
        ('吃了', '吃', '看了', '看'),
        ('走了', '走', '跑了', '跑'),
        ('写了', '写', '读了', '读'),
        ('说过', '说', '做过', '做'),
        ('打开', '开', '关上', '关'),
        ('进来', '进', '出去', '出'),

        # ===== 职业-场所 (6对) =====
        ('医生', '医院', '老师', '学校'),
        ('厨师', '厨房', '演员', '舞台'),
        ('法官', '法院', '牧师', '教堂'),
        ('士兵', '军队', '水手', '海军'),
        ('农民', '农田', '工人', '工厂'),
        ('作家', '书房', '画家', '画室'),

        # ===== 国家-语言 (5对) =====
        ('英语', '英国', '法国', '法语'),
        ('英语', '英国', '德国', '德语'),
        ('英语', '英国', '日本', '日语'),
        ('英语', '英国', '韩国', '韩语'),
        ('英语', '英国', '俄罗斯', '俄语'),

        # ===== 作家-作品 (6对) [原生中文] =====
        ('鲁迅', '呐喊', '老舍', '骆驼祥子'),
        ('曹雪芹', '红楼梦', '吴承恩', '西游记'),
        ('罗贯中', '三国演义', '施耐庵', '水浒传'),
        ('李白', '静夜思', '杜甫', '春望'),
        ('金庸', '射雕英雄传', '古龙', '多情剑客'),
        ('莫言', '红高粱', '余华', '活着'),

        # ===== 部分-整体 (5对) =====
        ('车轮', '汽车', '机翼', '飞机'),
        ('键盘', '电脑', '屏幕', '电视'),
        ('手指', '手', '脚趾', '脚'),
        ('门', '房子', '窗', '房间'),
        ('花瓣', '花', '叶', '树'),
    ]

    # 构建反向索引（idx -> word）
    idx2word = {i: w for w, i in word2idx.items()}

    results = []
    correct = 0
    total = 0

    for a, b, c, expected in analogy_tests:
        if not all(w in word2idx for w in [a, b, c]):
            continue

        total += 1
        vec = embeddings[word2idx[a]] - embeddings[word2idx[b]] + embeddings[word2idx[c]]

        sims = np.dot(embeddings, vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-10)

        # 排除查询词本身，避免返回 a/b/c
        exclude_idx = {word2idx[w] for w in [a, b, c]}
        for idx in exclude_idx:
            sims[idx] = -1

        best_idx = np.argmax(sims)
        best_word = idx2word[best_idx]
        best_sim = float(sims[best_idx])

        is_correct = (best_word == expected)
        if is_correct:
            correct += 1
        results.append({
            'a': a, 'b': b, 'c': c, 'expected': expected,
            'predicted': best_word,
            'similarity': best_sim,
            'correct': is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    return results, accuracy


def evaluate_all_versions():
    """评估所有版本"""
    print('='*50)
    print('词向量质量评估')
    print('='*50)

    versions = {
        'NumPy': 'numpy_version/output',
        'PyTorch': 'torch_version/output',
        'Gensim': 'gensim_version/output',
    }

    chinese_dir = 'chinese_wiki_full_output'

    all_results = {}

    # ===== 英文模型评估 =====
    for name, out_dir in versions.items():
        print(f'\n--- {name} 版本 ---')

        word2idx, embeddings = load_vocab_and_embeddings(out_dir)
        if word2idx is None:
            print(f'  跳过（输出文件不存在）')
            continue

        print(f'  词表大小: {len(word2idx)}')
        print(f'  向量维度: {embeddings.shape[1]}')

        # 词相似度评估
        sim_results, correlation = evaluate_word_similarity(embeddings, word2idx)
        print(f'  词相似度相关系数: {correlation:.4f}' if correlation else '  词相似度: 数据不足')

        # 词类比评估
        analogy_results, accuracy = evaluate_word_analogy(embeddings, word2idx)
        print(f'  词类比准确率: {accuracy:.2%} ({int(accuracy*len(analogy_results))}/{len(analogy_results)})')

        # 相似词测试
        test_words = ['king', 'computer', 'university', 'dog']
        print(f'  相似词测试:')
        for word in test_words:
            similar = find_similar_words(word, embeddings, word2idx, top_k=3)
            if similar:
                words_str = ', '.join([f'{w}({s:.3f})' for w, s in similar])
                print(f'    {word} → {words_str}')

        all_results[name] = {
            'vocab_size': len(word2idx),
            'embedding_dim': int(embeddings.shape[1]),
            'similarity_correlation': float(correlation) if correlation else None,
            'analogy_accuracy': float(accuracy),
            'analogy_results': analogy_results,
            'similarity_results': sim_results,
        }

    # ===== 中文模型评估（使用中文测试集）=====
    print(f'\n--- Chinese 版本 ---')

    word2idx, embeddings = load_vocab_and_embeddings(chinese_dir)
    if word2idx is not None:
        print(f'  词表大小: {len(word2idx)}')
        print(f'  向量维度: {embeddings.shape[1]}')

        # 中文词相似度评估
        sim_results, correlation = evaluate_chinese_similarity(embeddings, word2idx)
        print(f'  [中文] 词相似度相关系数: {correlation:.4f}' if correlation else '  [中文] 词相似度: 数据不足')
        print(f'    有效词对: {len(sim_results)}/35')

        # 中文词类比评估
        analogy_results, accuracy = evaluate_chinese_analogy(embeddings, word2idx)
        total = len(analogy_results)
        correct = sum(1 for r in analogy_results if r['correct'])
        print(f'  [中文] 词类比准确率: {accuracy:.2%} ({correct}/{total})')

        # 中文相似词测试
        test_words = ['国王', '计算机', '大学', '狗', '中国', '学习']
        print(f'  [中文] 相似词测试:')
        for word in test_words:
            similar = find_similar_words(word, embeddings, word2idx, top_k=5)
            if similar:
                words_str = ', '.join([f'{w}({s:.3f})' for w, s in similar])
                print(f'    {word} → {words_str}')

        all_results['Chinese'] = {
            'vocab_size': len(word2idx),
            'embedding_dim': int(embeddings.shape[1]),
            'similarity_correlation': float(correlation) if correlation else None,
            'analogy_accuracy': float(accuracy),
            'analogy_results': analogy_results,
            'similarity_results': sim_results,
        }
    else:
        print(f'  跳过（输出文件不存在）')

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'quality_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印对比表
    print('\n' + '='*60)
    print('质量评估对比总结')
    print('='*60)
    print(f'{"版本":<10} {"词表":<8} {"维度":<6} {"相似度相关":<12} {"类比准确率":<10}')
    print('-'*50)
    for name, r in all_results.items():
        corr = f'{r["similarity_correlation"]:.4f}' if r['similarity_correlation'] else 'N/A'
        acc = f'{r["analogy_accuracy"]:.2%}'
        print(f'{name:<10} {r["vocab_size"]:<8} {r["embedding_dim"]:<6} {corr:<12} {acc:<10}')

    print(f'\n[质量评估] 结果已保存到 {RESULTS_DIR}/quality_eval.json')
    return all_results


if __name__ == '__main__':
    evaluate_all_versions()
