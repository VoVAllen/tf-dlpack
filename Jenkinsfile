#!/usr/bin/env groovy
def init_git() {
  sh "rm -rf *"
  checkout scm
  sh "git submodule update --recursive --init"
}

pipeline {
  agent any
  stages {
    stage("Lint Check") {
      agent { 
        docker { 
          image "dgllib/tfdlpack-ci-gpu" 
          args "--runtime nvidia"
        } 
      }
      steps {
        init_git()
        sh "bash tests/scripts/task_lint_cpp.sh"
        sh "bash tests/scripts/task_lint_python.sh"
      }
      post {
        always {
          cleanWs disableDeferredWipeout: true, deleteDirs: true
        }
      }
    }

    stage("Build and Test") {
      agent { 
        docker { 
          image "dgllib/tfdlpack-ci-gpu" 
          args "--runtime nvidia"
        } 
      }
      steps {
        init_git()
        sh "bash tests/scripts/task_build.sh"
        sh "python -m pytest tests"
      }
      post {
        always {
          cleanWs disableDeferredWipeout: true, deleteDirs: true
        }
      }
    }
  }
}
