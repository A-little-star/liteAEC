# 定义编译器和编译选项
CC = gcc
CFLAGS = -Wall -O2 -Iinclude -g -w  # -w 忽略所有warning信息，调试代码时务必删掉:
LDFLAGS = -lm

# 定义项目结构
MODULE_DIR = ./module
MODEL_DIR = ./model
LIBS_DIR = ./libs
INC_DIR = ./include
BUILD_DIR = ./build
BIN_DIR = ./bin

# 定义 libs 目录下的 .c 文件
LIBS_SRC = $(wildcard $(LIBS_DIR)/*.c)
# 定义 module 目录下的 .c 文件
MODULE_SRC = $(wildcard $(MODULE_DIR)/*.c)
# 定义 model 目录下的 .c 文件
MODEL_SRC = $(wildcard $(MODEL_DIR)/*.c)

# 获取对应的目标文件路径，分别替换为 build 目录下的 .o 文件
LIBS_OBJ = $(patsubst $(LIBS_DIR)/%.c,$(BUILD_DIR)/%.o,$(LIBS_SRC))
MODULE_OBJ = $(patsubst $(MODULE_DIR)/%.c,$(BUILD_DIR)/%.o,$(MODULE_SRC))  
MODEL_OBJ = $(patsubst $(MODEL_DIR)/%.c,$(BUILD_DIR)/%.o,$(MODEL_SRC))

# 合并两个目录下的目标文件列表
OBJ = $(LIBS_OBJ) $(MODULE_OBJ) $(MODEL_OBJ)

# 定义最终的可执行文件名
TARGET = $(BIN_DIR)/inference

# 默认目标
all: $(TARGET)

# 编译libs/下的所有文件，生成目标文件
$(BUILD_DIR)/%.o: $(LIBS_DIR)/%.c
	@mkdir -p $(BUILD_DIR)  # 创建 build/ 目录（如果不存在）
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

# 编译module/下的所有文件，生成目标文件
$(BUILD_DIR)/%.o: $(MODULE_DIR)/%.c
	@mkdir -p $(BUILD_DIR)  # 创建 build/ 目录（如果不存在）
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

# 编译model/下的所有文件，生成目标文件
$(BUILD_DIR)/%.o: $(MODEL_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

# 生成可执行文件
$(TARGET): $(OBJ)
	@echo '$(OBJ)'
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

run:
	$(BIN_DIR)/inference

# 启动 gdb 调试
debug: $(TARGET)
	gdb $(TARGET)

# 清理编译生成的文件
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# 声明伪目标，防止与文件名冲突
.PHONY: all clean