def get_task(name, crossword_file=None):
    if name == 'game24':
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    elif name == 'text':
        from tot.tasks.text import TextTask
        return TextTask()
    elif name == 'crosswords':
        from tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask(file=crossword_file) if crossword_file else MiniCrosswordsTask()
    else:
        raise NotImplementedError