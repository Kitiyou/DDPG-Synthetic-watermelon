window.watermelonAI = {

    skipping: false,
    lastLevelUpFrame: 0,  // 最后一次水果合二为一时间

    init: function() {
        // 固定mainLoop()前进时间
         cc.director.calculateDeltaTime = function(){cc.director._deltaTime = 1/60};

        // 更改render函数，跳步时不渲染
        var originalRender = cc.renderer.render;
        cc.renderer.render = function() {
            if (window.watermelonAI.skipping)
                return;
            originalRender.apply(this, arguments);
        }

        // 记录最后一次水果合二为一时间
        var gameFunctionInstance = __require("GameFunction").default.Instance
        var originalCLUF = gameFunctionInstance.createLevelUpFruit;
        gameFunctionInstance.createLevelUpFruit = function() {
            window.watermelonAI.lastLevelUpFrame = cc.director.getTotalFrames();
            originalCLUF.apply(this, arguments);
        }

    },
    skipSteps: function(minStepsNum) {
        this.skipping = true;

        // 跳过minStepsNum帧
        for (var t = 0; t < minStepsNum; t++)
            cc.director.mainLoop();

        // 若有水果合二为一则等待1秒
        var waitSteps;
        while ((waitSteps = this.lastLevelUpFrame + 60 - cc.director.getTotalFrames()) > 0)
            for (var t = 0; t < waitSteps; t++)
                cc.director.mainLoop();

        this.skipping = false;
    },
    restart: function() {
        cc.find("Canvas").getComponent("MainGameJS").RestartGame();
    },
    isOver: function() {
        return !window.__require("PlayerInfo").default.GameUpdateCtrl;
    },
    getScore: function() {
        return cc.find("Canvas/gameManager").getComponent("GameManager").score;
    },
    getBoxFruits: function() {
        return cc.find("Canvas/fruitNode").children.map(
            fruit => [fruit.getComponent("fruitData").fruitNumber, fruit.x, fruit.y]
        );
    },
    getNextFruit: function() {
        var targetFruit = cc.find("Canvas/gameManager").getComponent("GameFunction").targetFruit;
        return targetFruit ? targetFruit.getComponent("fruitData").fruitNumber : null;
    },
    dropFruit: function(x) {
        var inputController = cc.find("Canvas/inputNode").getComponent("InputController");
        inputController.onTouchStart({getLocation: () => new cc.Vec2(x + 360, 0)});
        inputController.onTouchEnd();
    },
    getLineY: function() {
        return cc.find("Canvas/lineNode/虚线").y;
    }

};

window.watermelonAI.init();
